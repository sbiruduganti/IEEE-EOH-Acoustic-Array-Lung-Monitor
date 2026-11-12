import torch
from pathlib import Path
from src.model_qat import ICBHIModelQAT
from brevitas.export.onnx.qonnx.manager import QONNXManager

def export_to_finn():
    # Target software folder
    base_dir = Path(__file__).resolve().parent
    checkpoint_path = base_dir / "checkpoints" / "best_icbhi_model.pth"
    output_onnx_path = base_dir / "checkpoints" / "finn_audio_backbone.onnx"

    print("Initializing QAT Model (4-bit Weights & Activations)...")
    model_qat = ICBHIModelQAT(num_classes=4, weight_bit_width=4, act_bit_width=4)

    # Load unquantized FP32 weights for warm startup
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        model_qat.load_state_dict(state_dict, strict=False)
        print("Successfully pre-loaded floating-point checkpoint into QAT architecture!")
    else:
        print("Warning: Checkpoint not found. Exporting uninitialized QAT model.")

    model_qat.eval()

    # Export SharedAudioBackbone to FINN-compatible QONNX
    print("Exporting FINN/QONNX Model Graph...")
    dummy_input = torch.randn(1, 1, 129, 126)
    
    # Export using Brevitas QONNXManager (FINN native ONNX format)
    QONNXManager.export(model_qat.backbone, args=dummy_input, export_path=str(output_onnx_path))
    print(f"FINN ONNX model exported successfully to: {output_onnx_path}")

if __name__ == "__main__":
    export_to_finn()