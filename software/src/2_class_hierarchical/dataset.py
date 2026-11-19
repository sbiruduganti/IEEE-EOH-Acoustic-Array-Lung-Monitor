import os
import glob
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from scipy.io import wavfile as scipy_wavfile
    from scipy.signal import resample_poly
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import torchaudio
    import torchaudio.functional as AF
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


TARGET_DIAGNOSES = {
    "Healthy": 0,
    "COPD": 1,
    "Pneumonia": 2,
    "URTI": 3,
}
DIAGNOSIS_NAMES = ["Healthy", "COPD", "Pneumonia", "URTI"]


class HardwareReproducibleSTFT256(nn.Module):
    """
    Bit-accurate software emulation of UltraScale LogiCORE FFT v9.1:
      - 4,000 Hz sample rate, N_FFT = 256, Hop = 64 (75% overlap)
      - Cross-faded circular padding to 8,384 samples (2.096s) -> 128 frames
      - Drops Nyquist bin 128 -> keeps bins 0..127 -> exact (1, 128, 128) tensor
      - Stage scaling [2, 2, 2, 2] (/256), Alpha-Max-Beta-Min magnitude,
        and Base-2 Log LUT floor(16 * log2(mag + 1)) -> UINT8 [0..255] scaled to [0, 1].
    """

    def __init__(self, n_fft: int = 256, hop_length: int = 64, fade_samples: int = 32):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fade_samples = fade_samples
        self.target_samples = (128 - 1) * hop_length + n_fft  # 8384 samples

        n = torch.arange(n_fft, dtype=torch.float32)
        hann = 0.5 * (1.0 - torch.cos(2.0 * math.pi * n / n_fft))
        self.register_buffer("hann_q15", torch.round(hann * 32767.0))

    @staticmethod
    def crossfade_circular_pad_1d(
        waveform: torch.Tensor, target_len: int, fade_samples: int = 32
    ) -> torch.Tensor:
        """
        Circularly repeats short respiratory cycles to `target_len` while cross-fading
        the wrap-around seam so step discontinuities do not create artificial crackles.
        """
        cur_len = waveform.numel()
        if cur_len == 0:
            return torch.zeros(target_len, dtype=waveform.dtype, device=waveform.device)
        if cur_len >= target_len:
            return waveform[:target_len]

        fade = min(fade_samples, cur_len // 4)
        if fade > 1:
            alpha = torch.linspace(0.0, 1.0, fade, dtype=waveform.dtype, device=waveform.device)
            blended_seam = waveform[-fade:] * (1.0 - alpha) + waveform[:fade] * alpha
            cycle = torch.cat([waveform[:-fade], blended_seam], dim=0)
        else:
            cycle = waveform

        repeats = math.ceil(target_len / cycle.numel())
        return cycle.repeat(repeats)[:target_len]

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        peak = waveform.abs().max().clamp(min=1e-5)
        waveform = (waveform / peak).clamp(-1.0, 0.999969)

        waveform = self.crossfade_circular_pad_1d(
            waveform, self.target_samples, fade_samples=self.fade_samples
        )

        x_q15 = torch.round(waveform * 32768.0)
        frames = x_q15.unfold(dimension=-1, size=self.n_fft, step=self.hop_length)
        windowed = torch.round((frames * self.hann_q15) / 32768.0)

        fft_c = torch.fft.rfft(windowed, n=self.n_fft, dim=-1)[:, :128]
        real_q15 = torch.round(fft_c.real / 256.0)
        imag_q15 = torch.round(fft_c.imag / 256.0)

        abs_r, abs_i = real_q15.abs(), imag_q15.abs()
        hi, lo = torch.maximum(abs_r, abs_i), torch.minimum(abs_r, abs_i)
        mag_q15 = hi + torch.floor(lo / 4.0) + torch.floor(lo / 8.0)

        log2_uint8 = torch.floor(16.0 * torch.log2(mag_q15.t() + 1.0)).clamp(0.0, 255.0)
        return (log2_uint8 / 255.0).unsqueeze(0)  # (1, 128, 128)


class ICBHIDataset(Dataset):
    """
    Loads ICBHI breath cycles for the 4 target clinical diagnoses (Healthy, COPD, Pneumonia, URTI),
    computes/caches hardware-reproducible (1, 128, 128) spectrograms, and exposes patient IDs
    (`patient_ids`) and 4-class labels (`labels`) for global StratifiedGroupKFold splitting.
    """

    def __init__(
        self,
        data_dir: str,
        demo_file: Optional[str] = None,
        diag_file: Optional[str] = None,
        sample_rate: int = 4000,
        cache_path: Optional[str] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.stft_transform = HardwareReproducibleSTFT256(n_fft=256, hop_length=64, fade_samples=32)

        if cache_path is None:
            cache_path = os.path.join(self.data_dir, "icbhi_4khz_128x128_xfadecircpad_cache.pt")
        self.cache_path = cache_path

        self.demographics_map = self._load_demographics(demo_file)
        self.diagnosis_map = self._load_diagnoses(diag_file)
        self.samples: List[Dict] = self._load_or_build_cache()

        # Exposed arrays for global StratifiedGroupKFold splitting
        self.patient_ids = np.array([s["pid"] for s in self.samples])
        self.labels = np.array([s["label"] for s in self.samples], dtype=np.int64)

    @staticmethod
    def _encode_coarse_demographics(
        age: Optional[float], sex_str: str, bmi: Optional[float]
    ) -> List[float]:
        """
        Encodes patient demographics into 6 coarse clinical features to prevent
        exact continuous (age, bmi) values from acting as a patient ID shortcut:
          [age_pediatric (<18), age_adult (18-60), age_senior (>60),
           sex (1=M, 0=F, 0.5=NA), bmi_bucket (0=<25, 0.5=25-30, 1=>=30), demo_valid]
        """
        if age is not None and age > 0:
            age_ped = 1.0 if age < 18.0 else 0.0
            age_adult = 1.0 if 18.0 <= age <= 60.0 else 0.0
            age_senior = 1.0 if age > 60.0 else 0.0
            age_ok = 1.0
        else:
            age_ped, age_adult, age_senior, age_ok = 0.0, 0.0, 0.0, 0.0

        s_up = sex_str.upper()
        sex = 1.0 if s_up == "M" else (0.0 if s_up == "F" else 0.5)

        if bmi is not None and bmi > 0:
            if bmi < 25.0:
                bmi_bucket = 0.0
            elif bmi < 30.0:
                bmi_bucket = 0.5
            else:
                bmi_bucket = 1.0
            bmi_ok = 1.0
        else:
            bmi_bucket, bmi_ok = 0.0, 0.0

        demo_valid = 1.0 if (age_ok > 0.0 or bmi_ok > 0.0) else 0.0
        return [age_ped, age_adult, age_senior, sex, bmi_bucket, demo_valid]

    def _load_demographics(self, demo_path: Optional[str]) -> Dict[str, List[float]]:
        demo_dict: Dict[str, List[float]] = {}
        if demo_path is None:
            for cand in (
                os.path.join(self.data_dir, "ICBHI_challenge_demographics.txt"),
                os.path.join(self.data_dir, "demographic_info.txt"),
                os.path.join(os.path.dirname(self.data_dir), "ICBHI_challenge_demographics.txt"),
                os.path.join(os.path.dirname(self.data_dir), "demographic_info.txt"),
                "data/ICBHI_challenge_demographics.txt",
                "data/demographic_info.txt",
            ):
                if os.path.exists(cand):
                    demo_path = cand
                    break

        if not demo_path or not os.path.exists(demo_path):
            print("[ICBHIDataset] Warning: Demographics file not found; defaulting to zeros.")
            return demo_dict

        with open(demo_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4 or parts[0].lower() == "patient":
                    continue
                pid = str(int(parts[0])) if parts[0].isdigit() else parts[0]

                age: Optional[float] = None
                if parts[1] != "NA":
                    try:
                        age = float(parts[1])
                    except ValueError:
                        age = None

                sex_str = parts[2]

                bmi: Optional[float] = None
                if parts[3] != "NA":
                    try:
                        bmi = float(parts[3])
                    except ValueError:
                        bmi = None
                elif len(parts) >= 6 and parts[4] != "NA" and parts[5] != "NA":
                    # 6-column demographic_info.txt fallback: weight (kg) / (height (cm) / 100)^2
                    try:
                        w_kg = float(parts[4])
                        h_m = float(parts[5]) / 100.0
                        if h_m > 0:
                            bmi = w_kg / (h_m ** 2)
                    except ValueError:
                        bmi = None

                demo_dict[pid] = self._encode_coarse_demographics(age, sex_str, bmi)
        return demo_dict

    def _load_diagnoses(self, diag_path: Optional[str]) -> Dict[str, str]:
        diag_dict: Dict[str, str] = {}
        if diag_path is None:
            for cand in (
                os.path.join(self.data_dir, "ICBHI_challenge_diagnosis.txt"),
                os.path.join(self.data_dir, "patient_diagnosis.csv"),
                os.path.join(os.path.dirname(self.data_dir), "ICBHI_challenge_diagnosis.txt"),
                os.path.join(os.path.dirname(self.data_dir), "patient_diagnosis.csv"),
                "data/ICBHI_challenge_diagnosis.txt",
                "data/patient_diagnosis.csv",
            ):
                if os.path.exists(cand):
                    diag_path = cand
                    break

        if not diag_path or not os.path.exists(diag_path):
            raise FileNotFoundError(
                f"Could not find diagnosis file in {self.data_dir}. Pass --diag_file explicitly."
            )

        with open(diag_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().replace(",", " ").split()
                if len(parts) >= 2:
                    pid = str(int(parts[0])) if parts[0].isdigit() else parts[0]
                    diag_dict[pid] = parts[1]
        return diag_dict

    def _load_audio_4khz(self, wav_path: str) -> torch.Tensor:
        if SCIPY_AVAILABLE:
            orig_sr, data = scipy_wavfile.read(wav_path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = data.astype(np.float32)
            max_abs = np.max(np.abs(data))
            if max_abs > 0:
                data = data / max_abs
            if orig_sr != self.sample_rate:
                gcd = math.gcd(int(orig_sr), int(self.sample_rate))
                up = self.sample_rate // gcd
                down = int(orig_sr) // gcd
                data = resample_poly(data, up, down).astype(np.float32)
            return torch.from_numpy(data).unsqueeze(0)

        if TORCHAUDIO_AVAILABLE:
            waveform, orig_sr = torchaudio.load(wav_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if orig_sr != self.sample_rate:
                waveform = AF.resample(
                    waveform,
                    orig_freq=orig_sr,
                    new_freq=self.sample_rate,
                    lowpass_filter_width=64,
                    rolloff=0.90,
                    resampling_method="sinc_interp_kaiser",
                    beta=8.6,
                )
            return waveform

        raise RuntimeError("Neither scipy nor torchaudio is installed to read .wav files.")

    def _load_or_build_cache(self) -> List[Dict]:
        if os.path.exists(self.cache_path):
            print(f"[ICBHIDataset] Loading precomputed cache from {self.cache_path}...")
            return torch.load(self.cache_path, map_location="cpu", weights_only=False)

        txt_files = sorted(glob.glob(os.path.join(self.data_dir, "**", "*.txt"), recursive=True))
        records: List[Dict] = []

        print(f"[ICBHIDataset] Building 4kHz cross-faded circpad STFT cache across {len(txt_files)} files...")
        for txt_path in txt_files:
            base = os.path.basename(txt_path)
            if any(k in base.lower() for k in ("demographics", "diagnosis", "filename", "events", "readme")):
                continue
            wav_path = os.path.splitext(txt_path)[0] + ".wav"
            if not os.path.exists(wav_path):
                continue

            raw_pid = os.path.splitext(base)[0].split("_")[0]
            pid = str(int(raw_pid)) if raw_pid.isdigit() else raw_pid
            diag_name = self.diagnosis_map.get(pid, None)
            if diag_name not in TARGET_DIAGNOSES:
                continue

            waveform = self._load_audio_4khz(wav_path)
            demo_vec = torch.tensor(
                self.demographics_map.get(pid, [0.0, 0.0, 0.0, 0.5, 0.0, 0.0]),
                dtype=torch.float32,
            )

            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    cols = line.strip().split()
                    if len(cols) < 4:
                        continue
                    start_t, end_t = float(cols[0]), float(cols[1])
                    s_idx = int(start_t * self.sample_rate)
                    e_idx = int(end_t * self.sample_rate)
                    cycle_wav = waveform[:, s_idx:e_idx]
                    if cycle_wav.numel() < 400:  # skip corrupted <0.1s slices
                        continue
                    spec_float = self.stft_transform(cycle_wav)
                    spec_uint8 = torch.round(spec_float * 255.0).to(torch.uint8)
                    records.append(
                        {
                            "pid": pid,
                            "diagnosis_name": diag_name,
                            "label": TARGET_DIAGNOSES[diag_name],
                            "spec_uint8": spec_uint8,
                            "demographics": demo_vec,
                        }
                    )

        if len(records) > 0:
            os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
            torch.save(records, self.cache_path)
            print(f"[ICBHIDataset] Cached {len(records)} breath cycles to {self.cache_path}.")
        return records

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        spec_float = s["spec_uint8"].to(torch.float32) / 255.0
        return {
            "spectrogram": spec_float,          # (1, 128, 128)
            "demographics": s["demographics"],  # (6,)
            "label": torch.tensor(s["label"], dtype=torch.long),
            "pid": s["pid"],
        }