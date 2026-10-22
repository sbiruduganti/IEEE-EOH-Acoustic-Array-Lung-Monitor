import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio

class ICBHIDataset(Dataset):
    def __init__(self, data_dir, sample_rate=4000, max_len=4.0, n_mels=64):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * max_len) # 16,000 samples for 4.0s
        self.resamplers = {} # Cache resamplers by original sample rate (e.g., 44100, 10000)

        # Log-Mel Spectrogram bounded to clinical lung sound frequencies (50 Hz - 2000 Hz)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            hop_length=256,
            n_mels=n_mels,
            f_min=50.0,
            f_max=float(sample_rate // 2)
        )

        # Load demographic data (supports both ICBHI demographic_info.txt and custom format)
        self.demographics = {}
        for fname in ["filename_to_demographics.txt", "demographic_info.txt"]:
            demo_path = os.path.join(data_dir, fname)
            if os.path.exists(demo_path):
                df_demo = pd.read_csv(demo_path, sep=r"\s+|\t", engine="python", header=None, na_values="NA")
                for _, row in df_demo.iterrows():
                    pid = str(int(row[0])) if pd.notna(row[0]) and str(row[0]).isdigit() else str(row[0])
                    age = float(row[1]) / 100.0 if pd.notna(row[1]) else 0.60
                    sex = 1.0 if str(row[2]).strip().upper() == "M" else 0.0
                    # Calculate BMI from Adult BMI (col 3) or Child Weight/Height (cols 4, 5)
                    if pd.notna(row[3]):
                        bmi = float(row[3]) / 50.0
                    elif len(row) > 5 and pd.notna(row[4]) and pd.notna(row[5]) and float(row[5]) > 0:
                        bmi = (float(row[4]) / ((float(row[5]) / 100.0) ** 2)) / 50.0
                    else:
                        bmi = 0.50
                    self.demographics[pid] = [age, bmi, sex, 0.0, 0.0]
                break

        # Parse every annotated breathing cycle from the matching .txt files
        self.cycles = []
        wav_files = sorted([f[:-4] for f in os.listdir(data_dir) if f.endswith(".wav")])
        for file_id in wav_files:
            txt_path = os.path.join(data_dir, file_id + ".txt")
            wav_path = os.path.join(data_dir, file_id + ".wav")
            patient_id = file_id.split("_")[0]

            if os.path.exists(txt_path):
                df_cycles = pd.read_csv(txt_path, sep=r"\s+", header=None, names=["start", "end", "crackle", "wheeze"])
                for _, row in df_cycles.iterrows():
                    duration = float(row["end"]) - float(row["start"])
                    if duration < 0.3: # Screen out corrupted micro-cycles (< 0.3s)
                        continue
                    c, w = int(row["crackle"]), int(row["wheeze"])
                    # 0: Healthy, 1: Wheeze, 2: Crackle, 3: Both (matches app.py class order)
                    if c == 0 and w == 0:
                        label = 0
                    elif c == 0 and w == 1:
                        label = 1
                    elif c == 1 and w == 0:
                        label = 2
                    else:
                        label = 3
                    self.cycles.append((wav_path, patient_id, float(row["start"]), float(row["end"]), label))

    def __len__(self):
        return len(self.cycles)

    def __getitem__(self, idx):
        wav_path, patient_id, start_sec, end_sec, label = self.cycles[idx]
        waveform, sr = torchaudio.load(wav_path)

        # 1. Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 2. Slice exact respiratory cycle BEFORE resampling for speed
        start_frame = int(start_sec * sr)
        end_frame = min(int(end_sec * sr), waveform.shape[1])
        waveform = waveform[:, start_frame:end_frame]

        # 3. Cached resampling to 4000 Hz
        if sr != self.sample_rate:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = self.resamplers[sr](waveform)

        # 4. Zero-mean & peak-normalize cycle to screen out stethoscope gain differences
        waveform = waveform - waveform.mean()
        max_val = waveform.abs().max()
        if max_val > 1e-6:
            waveform = waveform / max_val

        # 5. Circular pad (tile) short cycles or trim long cycles to 16,000 samples (4.0s)
        cur_len = waveform.shape[1]
        if cur_len < self.target_samples:
            repeats = (self.target_samples // cur_len) + 1
            waveform = waveform.repeat(1, repeats)[:, :self.target_samples]
        else:
            waveform = waveform[:, :self.target_samples]

        # 6. Compute Log-Mel Spectrogram [1, 64, 63] and normalize to [0, 1] for 8-bit quantization
        mel_spec = self.mel_transform(waveform)
        log_mel = torch.log(mel_spec + 1e-6)
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)

        # 7. Retrieve patient demographics + replace slot 4 with normalized cycle duration
        demo_vec = list(self.demographics.get(patient_id, [0.6, 0.5, 1.0, 0.0, 0.0]))
        demo_vec[3] = min((end_sec - start_sec) / 10.0, 1.0) # Inject real breathing cycle length!
        demo_tensor = torch.tensor(demo_vec, dtype=torch.float32)

        return waveform, log_mel, demo_tensor, torch.tensor(label, dtype=torch.long), patient_id