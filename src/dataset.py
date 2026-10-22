import os
import glob
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
import torchaudio.transforms as T

class ICBHIDataset(Dataset):
    """
    This is the parser for the dataset from kaggle.
    It parses audio cycles, anotations, and patient demographic metadata.
    """

    def __init__(self, data_dir, demo_file=None, sample_rate=4000, target_duration=2.0):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * target_duration)

        #2d stft spectrogram transformer
        self.spectrogram_transform = T.Spectrogram(n_fft=256, hop_length=64)
        
        #initialize demographics and annotations next
        self.demographics = self._load_demographics(demo_file)
        self.samples = self._parse_annotations()

    def _load_demographics(self, demo_path):
        demo_dict = {}
        if demo_path is None or not os.path.exists(demo_path):
            print(f"Warning: Demographic file '{demo_path}' not found. Defaulting mask flags to 0.")
            return demo_dict
        
        with open(demo_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts)<6:
                    continue
                try:
                    patient_id = int(parts[0])

                except ValueError:
                    continue #skip header or malformed rows
                
                #age
                age_present = 1.0 if parts[1] != 'NA' else 0.0
                age = float(parts[1]) if age_present else 0.0

                #sex
                sex = 1.0 if parts[2] == 'M' else 0.0

                #bmi
                bmi_present = 0.0
                bmi = 0.0

                if parts[3] != 'NA':
                    bmi = float(parts[3])
                    bmi_present = 1.0
                elif parts[4] != 'NA' and parts[5] != 'NA':
                    try:
                        weight_kg = float(parts[4])
                        height_m = float(parts[5])/100.0
                        if height_m>0:
                            bmi = weight_kg / (height_m**2)
                            bmi_present = 1.0
                    except ValueError:
                        bmi = 0.0

                #5-element vector
                demo_dict[patient_id] = [age, sex, bmi, age_present, bmi_present]
        return demo_dict
    
    def _parse_annotations(self):
        samples = []

        #Find all .txt files inside data_dir (including subdirectories)
        txt_files = glob.glob(os.path.join(self.data_dir, "**", "*.txt"), recursive=True)

        for txt_path in txt_files:
            #match text fileto audio .wav file
            wav_path = txt_path.replace(".txt", ".wav")
            if not os.path.exists(wav_path):
                continue
            
            filename = os.path.basename(txt_path)
            try:
                #extract patient id
                patient_id = int(filename.split("_")[0])
            except ValueError:
                continue
            
            #read line by line
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts)<4:
                        continue
                    
                    start_t = float(parts[0])
                    end_t = float(parts[1])
                    crackle = int(parts[2])
                    wheeze = int(parts[3])

                    #Map flags to 4 classes

                    if crackle == 0 and wheeze == 0:
                        label = 0 #normal
                    elif crackle == 1 and wheeze == 0:
                        label = 1 #crackle
                    elif crackle == 0 and wheeze == 1:
                        label = 2 #wheeze
                    else:
                        label = 3 #both
                    
                    #save dictionary entry for breath cycle
                    samples.append({
                        "wav_path": wav_path,
                        "start_t": start_t,
                        "end_t": end_t,
                        "label": label,
                        "patient_id": patient_id
                    })
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        #load audio clips
        waveform, sr = torchaudio.load(sample["wav_path"])
        
        #resample to target sample rate if needed
        if sr != self.sample_rate:
            resample = T.Resample(sr, self.sample_rate)
            waveform = resample(waveform)
        
        # convert mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        #slice audio to specific breath cycle interval
        start_sample = int(sample["start_t"]*self.sample_rate)
        end_sample = int(sample["end_t"] * self.sample_rate)
        cycle_wave = waveform[:, start_sample:end_sample]

        #edge case safeguard for empty subslices
        if cycle_wave.numel() == 0:
            cycle_wave = torch.zeros((1, self.target_length), dtype=torch.float32)

        #pad or trim waveform to standard target duration
        if cycle_wave.shape[1] < self.target_length:
            pad_amount = self.target_length - cycle_wave.shape[1]
            cycle_wave = torch.nn.functional.pad(cycle_wave, (0, pad_amount))
        else:
            cycle_wave = cycle_wave[:, :self.target_length]
        
        #convert 1d waveform to 2d spectrogram image
        spectrogram = self.spectrogram_transform(cycle_wave)

        #4 fetch 5 element demographic vector [age, sex, smi, age_mask, bmi_mask]
        demo_vec = self.demographics.get(sample["patient_id"], [0.0, 0.0, 0.0, 0.0, 0.0])
        demo_tensor = torch.tensor(demo_vec, dtype=torch.float32)

        label_tensor = torch.tensor(sample["label"], dtype=torch.long)

        return {
            "spectrogram": spectrogram,
            "demographics": demo_tensor,
            "label": label_tensor
        }




