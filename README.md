# 🫁 8-Channel Acoustic Array Lung Monitor (MVP)

Interactive Python/Streamlit prototype simulating multi-channel respiratory sound processing, 2D STFT spectrogram generation, and multimodal PyTorch CNN classification.

## Features
- **8-Channel Audio Simulation:** Synthetic breathing audio with injected 400 Hz pathological wheeze on specific array channels.
- **2D STFT Spectrogram Transformation:** Audio signal conversion using `torchaudio.transforms.Spectrogram`.
- **Multi-Channel PyTorch CNN:** Custom 2D CNN architecture accepting 8-channel spectrogram tensors.
- **Multimodal Diagnostic Risk Score:** Combines CNN feature probabilities with patient age and smoking risk factors.
- **Interactive UI:** Streamlit dashboard with dynamic parameter controls and STFT intensity visualizations.

## Quickstart
```bash
pip install torch torchaudio numpy matplotlib streamlit
streamlit run app.py