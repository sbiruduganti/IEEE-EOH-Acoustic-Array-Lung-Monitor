import os
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio


class ICBHIDataset(Dataset):
    def __init__(self, data_dir, sample_rate=4000, max_len=4.0, n_mels=64):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * max_len)  # 16,000 samples (4.0s)
        self.cache_path = os.path.join(data_dir, "icbhi_cache_4khz_v2.pt")

        if os.path.exists(self.cache_path):
            print(f"[Dataset] Loading pre-computed 4000 Hz cache from {self.cache_path} ...", flush=True)
            cache = torch.load(self.cache_path, weights_only=False)
            self.wave_env = cache["wave_env"]      # [N, 2, 256] compact 1D transient envelope
            self.specs = cache["specs"]            # [N, 3, 64, 64] Log-Mel + Delta + Delta-Delta
            self.demos = cache["demos"]            # [N, 5]
            self.labels = cache["labels"]          # [N]
            self.patient_ids = cache["patient_ids"]
            return

        print("[Dataset] Building 4000 Hz feature cache (runs once, takes ~45s)...", flush=True)
        resamplers = {}
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=512, hop_length=251, n_mels=n_mels, f_min=50.0, f_max=2000.0
        )
        delta_transform = torchaudio.transforms.ComputeDeltas()

        # Load demographic data
        demographics = {}
        for fname in ["filename_to_demographics.txt", "demographic_info.txt"]:
            demo_path = os.path.join(data_dir, fname)
            if os.path.exists(demo_path):
                df_demo = pd.read_csv(demo_path, sep=r"\s+|\t", engine="python", header=None, na_values="NA")
                for _, row in df_demo.iterrows():
                    pid = str(int(row[0])) if pd.notna(row[0]) and str(row[0]).isdigit() else str(row[0])
                    age = float(row[1]) / 100.0 if pd.notna(row[1]) else 0.60
                    sex = 1.0 if str(row[2]).strip().upper() == "M" else 0.0
                    if pd.notna(row[3]):
                        bmi = float(row[3]) / 50.0
                    elif len(row) > 5 and pd.notna(row[4]) and pd.notna(row[5]) and float(row[5]) > 0:
                        bmi = (float(row[4]) / ((float(row[5]) / 100.0) ** 2)) / 50.0
                    else:
                        bmi = 0.50
                    demographics[pid] = [age, bmi, sex, 0.0, 0.0]
                break

        # Location encoding map (chest site strongly affects background breath sounds)
        loc_map = {"Tc": 0.1, "Al": 0.25, "Ar": 0.4, "Pl": 0.55, "Pr": 0.7, "Ll": 0.85, "Lr": 1.0}

        wave_env_list, spec_list, demo_list, label_list, pid_list = [], [], [], [], []
        wav_files = sorted([f[:-4] for f in os.listdir(data_dir) if f.endswith(".wav")])

        for f_idx, file_id in enumerate(wav_files):
            txt_path = os.path.join(data_dir, file_id + ".txt")
            wav_path = os.path.join(data_dir, file_id + ".wav")
            if not os.path.exists(txt_path):
                continue

            parts = file_id.split("_")
            patient_id = parts[0]
            chest_loc = loc_map.get(parts[2], 0.5) if len(parts) > 2 else 0.5

            # Read full .wav file ONCE per recording (920 reads instead of 6,898 reads)
            audio_np, sr = sf.read(wav_path, dtype="float32", always_2d=True)
            full_wave = torch.from_numpy(audio_np.T)
            if full_wave.shape[0] > 1:
                full_wave = torch.mean(full_wave, dim=0, keepdim=True)

            if sr != sample_rate:
                if sr not in resamplers:
                    resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                full_wave = resamplers[sr](full_wave)

            df_cycles = pd.read_csv(txt_path, sep=r"\s+", header=None, names=["start", "end", "crackle", "wheeze"])
            for _, row in df_cycles.iterrows():
                start_sec, end_sec = float(row["start"]), float(row["end"])
                duration = end_sec - start_sec
                if duration < 0.3:
                    continue

                start_idx = max(0, int(start_sec * sample_rate))
                end_idx = min(int(end_sec * sample_rate), full_wave.shape[1])
                if end_idx - start_idx < int(0.3 * sample_rate):
                    continue

                wave = full_wave[:, start_idx:end_idx]
                wave = wave - wave.mean()
                std = wave.std()
                if std < 1e-5:
                    continue
                wave = wave / (std * 3.0)  # Standardized RMS normalization (preserves spike contrast!)
                wave = torch.clamp(wave, -1.0, 1.0)

                cur_len = wave.shape[1]
                if cur_len < self.target_samples:
                    repeats = (self.target_samples // cur_len) + 1
                    wave = wave.repeat(1, repeats)[:, : self.target_samples]
                else:
                    wave = wave[:, : self.target_samples]

                # 1. Extract fast 1D transient crackle envelope [2, 256]:
                # Channel 0: Max absolute peak in each 62-sample (15.6ms) window
                # Channel 1: Mean absolute energy (crest difference catches impulsive crackles)
                w_frames = wave[:, :15872].view(1, 256, 62).abs()
                peak_env = w_frames.max(dim=2)[0]
                mean_env = w_frames.mean(dim=2)
                env_1d = torch.cat([peak_env, peak_env - mean_env], dim=0)  # [2, 256]

                # 2. Extract 3-Channel 2D Spectrogram [3, 64, 64]: Log-Mel + Delta + Delta-Delta
                mel = mel_transform(wave)[:, :, :64]  # [1, 64, 64]
                log_mel = (10.0 * torch.log10(mel + 1e-6) + 60.0) / 60.0
                log_mel = torch.clamp(log_mel, 0.0, 1.0)
                d1 = torch.clamp(delta_transform(log_mel) * 2.0 + 0.5, 0.0, 1.0)
                d2 = torch.clamp(delta_transform(d1) * 2.0 + 0.5, 0.0, 1.0)
                spec_3ch = torch.cat([log_mel, d1, d2], dim=0)  # [3, 64, 64]

                c, w = int(row["crackle"]), int(row["wheeze"])
                label = 0 if (c == 0 and w == 0) else (1 if (c == 0 and w == 1) else (2 if (c == 1 and w == 0) else 3))

                demo_vec = list(demographics.get(patient_id, [0.6, 0.5, 1.0, 0.0, 0.0]))
                demo_vec[3] = min(duration / 8.0, 1.0)  # Breathing cycle duration
                demo_vec[4] = chest_loc                 # Stethoscope chest location

                wave_env_list.append(env_1d.to(torch.float16))
                spec_list.append(spec_3ch.to(torch.float16))
                demo_list.append(torch.tensor(demo_vec, dtype=torch.float32))
                label_list.append(label)
                pid_list.append(patient_id)

            if (f_idx + 1) % 200 == 0 or (f_idx + 1) == len(wav_files):
                print(f"  Processed {f_idx+1}/{len(wav_files)} recordings...", flush=True)

        self.wave_env = torch.stack(wave_env_list)
        self.specs = torch.stack(spec_list)
        self.demos = torch.stack(demo_list)
        self.labels = torch.tensor(label_list, dtype=torch.long)
        self.patient_ids = pid_list

        torch.save(
            {
                "wave_env": self.wave_env,
                "specs": self.specs,
                "demos": self.demos,
                "labels": self.labels,
                "patient_ids": self.patient_ids,
            },
            self.cache_path,
        )
        print(f"[Dataset] Saved {len(self.labels)} cycles to {self.cache_path}!", flush=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.wave_env[idx].float(),
            self.specs[idx].float(),
            self.demos[idx],
            self.labels[idx],
            self.patient_ids[idx],
        )