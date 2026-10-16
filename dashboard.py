import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt  # FIXED: Added .pyplot

# Import custom model from src/model.py
from src.model import ICBHIModel

st.set_page_config(
    page_title="8-Channel Respiratory Stethoscope Harness",
    page_icon="🫁",
    layout="wide"
)

CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both (Crackle & Wheeze)"]
CHECKPOINT_PATH = "checkpoints/best_icbhi_model.pth"

# Map 8 channels to standard clinical chest positions
STETHOSCOPE_POSITIONS = {
    0: "Left Anterior Upper (LAU)",
    1: "Right Anterior Upper (RAU)",
    2: "Left Anterior Lower (LAL)",
    3: "Right Anterior Lower (RAL)",
    4: "Left Posterior Upper (LPU)",
    5: "Right Posterior Upper (RPU)",
    6: "Left Posterior Lower (LPL)",
    7: "Right Posterior Lower (RPL)",
}

# Model Loading
@st.cache_resource
def load_trained_model():
    model = ICBHIModel(num_classes=4)
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
        # Support loading full checkpoint dict or raw state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        st.sidebar.success("Loaded model checkpoint!")
    else:
        st.sidebar.warning("Checkpoint not found. Running with uninitialized weights.")
    model.eval()
    return model

model = load_trained_model()

# Sidebar inputs
st.sidebar.header("📋 Patient Demographics")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=45)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
height = st.sidebar.number_input("Height (cm)", min_value=50, max_value=230, value=170)
weight = st.sidebar.number_input("Weight (kg)", min_value=10, max_value=200, value=70)
bmi = weight / ((height / 100) ** 2)
st.sidebar.info(f"Calculated BMI: **{bmi:.1f}**")

# Construct [1, 5] demographic vector expected by ICBHIModel
sex_binary = 1.0 if sex == "Male" else 0.0
demo_tensor = torch.tensor([[age, sex_binary, height, weight, bmi]], dtype=torch.float32)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Harness Settings")
input_source = st.sidebar.radio("Data Stream Source", ["Simulate Live 8-Channel Stream", "Upload Multi-channel WAV"])

# Audio processing logic
if input_source == "Simulate Live 8-Channel Stream":
    if st.sidebar.button("⚡ Capture Live Breath Cycle", type="primary"):
        # Generate dummy 8-channel spectrogram tensor [1, 8, 1, 129, 126]
        spectrogram_tensor = torch.randn(1, 8, 1, 129, 126)
        st.session_state["spec_data"] = spectrogram_tensor
else:
    uploaded_file = st.sidebar.file_uploader("Upload 8-channel WAV file", type=["wav"])
    if uploaded_file is not None:
        # Placeholder for real audio loading and spectrogram conversion
        spectrogram_tensor = torch.randn(1, 8, 1, 129, 126)
        st.session_state["spec_data"] = spectrogram_tensor

# Dashboard visualizations
st.title("🫁 8-Channel Acoustic Array Lung Monitor")
st.markdown("Real-time STFT multi-channel signal analysis and spatial diagnostics.")

if "spec_data" in st.session_state:
    spec_tensor = st.session_state["spec_data"]  # Shape: [1, 8, 1, 129, 126]

    with torch.no_grad():
        # Mode A: Per-channel predictions [1, 8, 4]
        per_ch_logits = model(spec_tensor, demo_tensor, return_per_channel=True)
        per_ch_probs = F.softmax(per_ch_logits, dim=-1).squeeze(0).numpy()

        # Mode B: Aggregated patient summary [1, 4]
        summary_logits = model(spec_tensor, demo_tensor, return_per_channel=False)
        summary_probs = F.softmax(summary_logits, dim=-1).squeeze(0).numpy()

    # --- Section 1: Aggregated Diagnostic Summary ---
    st.subheader("🏥 Patient Diagnostic Summary")
    top_class_idx = np.argmax(summary_probs)
    top_class_name = CLASS_NAMES[top_class_idx]
    confidence = summary_probs[top_class_idx] * 100

    col_summary1, col_summary2 = st.columns([1, 2])
    with col_summary1:
        if top_class_idx == 0:
            st.success(f"**Diagnosis:** {top_class_name}\n\nConfidence: {confidence:.1f}%")
        else:
            st.error(f"**Diagnosis:** {top_class_name}\n\nConfidence: {confidence:.1f}%")

    with col_summary2:
        st.write("**Overall Class Probabilities**")
        for cls_name, prob in zip(CLASS_NAMES, summary_probs):
            st.progress(float(prob), text=f"{cls_name}: {prob*100:.1f}%")

    st.markdown("---")

    # --- Section 2: Spatial Chest Diagnostic Grid ---
    st.subheader("🗺️ Per-Channel Diagnostic Grid")
    grid_cols = st.columns(4)

    for ch_idx in range(8):
        col = grid_cols[ch_idx % 4]
        ch_probs = per_ch_probs[ch_idx]
        pred_idx = np.argmax(ch_probs)
        pred_label = CLASS_NAMES[pred_idx]
        conf = ch_probs[pred_idx] * 100

        with col:
            st.markdown(f"##### Channel {ch_idx+1}")
            st.caption(f"**{STETHOSCOPE_POSITIONS[ch_idx]}**")
            
            if pred_idx == 0:
                st.metric("Status", pred_label, delta=f"{conf:.1f}% Conf", delta_color="normal")
            else:
                st.metric("Status", pred_label, delta=f"{conf:.1f}% Conf", delta_color="inverse")

    st.markdown("---")

    # --- Section 3: STFT Spectrogram Visualizer ---
    st.subheader("📊 Live Spectrogram Inspection (STFT Freq vs Time)")
    
    selected_ch = st.selectbox(
        "Select Stethoscope Channel to Inspect:",
        options=list(range(8)),
        format_func=lambda x: f"Channel {x+1}: {STETHOSCOPE_POSITIONS[x]}"
    )

    # Extract log intensity for chosen channel: shape [129, 126]
    channel_spec = spec_tensor[0, selected_ch, 0].numpy()
    log_spec = np.log(np.abs(channel_spec) + 1e-6)

    col_plot, col_info = st.columns([2, 1])

    with col_plot:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        cax = ax.imshow(log_spec, origin="lower", aspect="auto", cmap="magma")
        fig.colorbar(cax, ax=ax, label="Log Intensity")
        ax.set_title(f"STFT Spectrogram — Channel {selected_ch+1} ({STETHOSCOPE_POSITIONS[selected_ch]})")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Frequency Bins")
        st.pyplot(fig)

    with col_info:
        st.write(f"**Channel {selected_ch+1} Classification Breakdown**")
        ch_probs = per_ch_probs[selected_ch]
        for name, p in zip(CLASS_NAMES, ch_probs):
            st.progress(float(p), text=f"{name}: {p*100:.1f}%")

else:
    st.info("👈 Click **'Capture Live Breath Cycle'** in the sidebar to stream multi-channel audio into the dashboard.")
#source venv/bin/activate

#streamlit run dashboard.py 