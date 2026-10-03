import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
#generate single channel signal
def generate_single_channel(duration_sec=2, sample_rate=4000):
    #time steps
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint = False)
    #simulate breathing sound via smooth low freq sine wave and slight noise
    breathing = 0.5 * np.sin(2 * np.pi * 1.5*t) + 0.1 * np.random.normal(size=t.shape)

    return t, breathing

#generate multi-channel signal as tensor for neural network
def generate_8_channel_array(duration_sec = 2, sample_rate=4000, pathology=False):
    t, _ = generate_single_channel(duration_sec, sample_rate)
    channels = [] #store 8 channels

    for ch in range(8):
        #generate a base sound
        _, base_sound = generate_single_channel(duration_sec, sample_rate)

        #if pathology active inject 400hz high frquency wheeze on ch 3 and 4
        if pathology and ch in [2, 3]:
            wheeze = 0.4 * np.sin(2 * np.pi * 400 * t)
            base_sound += wheeze

        channels.append(base_sound)

    return torch.tensor(np.array(channels), dtype=torch.float32)

#convert raw audio tensor to 2D frequency time spectorgrams with short time fourier transform stft

def convert_to_spectrograms(audio_tensor):
    #converts 1d raw audio waves to 2d stft spectrogram 8, freq, time

    #define spectrogram transform, size of fourier transform window, distance between windows
    spectrogram_transform = T.Spectrogram(n_fft=256, hop_length=64)

    spectrograms = spectrogram_transform(audio_tensor)
    return spectrograms

class MultiChannelAcousticCNN(nn.Module): #defines a trainable neural network
    def __init__(self):
        super().__init__() #constructor to declare and instantiate layers, call constuctor of nn.Module to set up
        #8 input channels for 8 stethoscopes
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1) #8 physical stethoscope channels, 16 learnable 2d kernels, kernels are 3x3, 1 pixel border to prevent edges from shrinking
        self.relu = nn.ReLU() #takes every number from the convolution and turns off negative
        self.pool = nn.MaxPool2d(2, 2) #keeps max from each 2x2 region

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8,8))
        self.fc = nn.Linear(32*8*8,2) #0 = normal, 1 = wheeze detected

    def forward(self, x):
        #input shape (batch, 8_channels, Fre_Bins, Time_Steps)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1) #flatten 2d feature map to 1d vector
        return self.fc(x)

    
def predict_with_demographics(acoustic_logits, age, is_smoker):
    #fuses acoustic features with patient risk factors to generate diagnostic probability
    #softmax raw cnn output to get acoustic probability
    probabilities = torch.softmax(acoustic_logits, dim=1)
    base_pathology_prob = probabilities[0][1].item()

    #apply risk weighting rules
    risk_factor = 1.0
    if age > 50:
        risk_factor += 0.15
    if is_smoker:
        risk_factor += 0.25

    final_score = min(base_pathology_prob * risk_factor, 1.0)
    return final_score

#streamlit dashboard interface
st.set_page_config(page_title="EOH Acoustic Array Monitor", layout="wide")
st.title("🫁 8-Channel Acoustic Array Lung Monitor")
st.markdown("Real-time STFT signal processing and multimodal CNN diagnostics.")

#Sidebar Controls
st.sidebar.header("Patient Demographics & Simulation Settings")
age = st.sidebar.slider("Patient Age", 18, 90, 45)
is_smoker = st.sidebar.checkbox("History of Smoking")
simulate_pathology = st.sidebar.toggle("Inject Pathological Wheeze (400 Hz)")


if st.sidebar.button("Run Acoustic Scan"):
    #generate and process audio
    raw_audio = generate_8_channel_array(pathology=simulate_pathology)
    spectrograms = convert_to_spectrograms(raw_audio)

    #2 run nn inference
    model = MultiChannelAcousticCNN()
    model.eval()
    with torch.no_grad():
        input_tensor = spectrograms.unsqueeze(0)
        logits = model(input_tensor)
    
    risk_score = predict_with_demographics(logits, age, is_smoker)

    #3 display diagnostic output
    st.subheader("Diagnostic Classification Results")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Calculated Disease Probability", value = f"{risk_score*100: .1f}%")
        if risk_score >0.5:
            st.error("⚠️️ ALERT: Pathological Wheeze / Crackle Detected")
        else:
            st.success("✅ Sound Profile: Healthy / Normal Breathing")
    
    with col2:
        #Display spectrogram heatmap of channel 3, the one with the injected wheeze
        fig, ax = plt.subplots(figsize = (6,3))
        log_spec = torch.log(spectrograms[2] + 1e-6).numpy()
        cax = ax.imshow(log_spec, origin ="lower", aspect="auto", cmap="magma")
        fig.colorbar(cax, ax=ax, label = "Log Intensity")
        ax.set_title("Channel 3 STFT Spectrogram (Freq vs Time)")
        ax.set_xlabel("Time Bins")
        ax.set_ylabel("Freq Bins")
        st.pyplot(fig)