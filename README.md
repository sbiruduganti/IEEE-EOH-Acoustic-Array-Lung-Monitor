# 8-Channel Acoustic Array Lung Monitor

IEEE UIUC project for Engineering Open House (EOH) 2027.

An 8-channel acoustic array that records lung sounds and classifies them on an FPGA. A custom ADC board digitizes all eight channels at the same time, and a quantized neural network running on an Opal Kelly XEM8305 (AMD Artix UltraScale+) flags abnormal sounds such as wheezes and crackles.

## System overview

```
8 audio channels -> ADC board -> FPGA (audio receiver + CNN accelerator) -> host PC
```

| Subsystem | Scope | Folder |
|---|---|---|
| Circuit design | Analog front end, 8-channel ADC, power, PCB | `circuit-design/` |
| FPGA | ADC interface, FINN neural network accelerator | not yet in repo |
| Software | Dataset pipeline, model training, quantization, demo dashboards | `software/` |

## Repository layout

```
circuit-design/       ADC board: KiCad files, BOM, design choices
software/
  app.py              Streamlit demo with simulated 8-channel audio
  dashboard.py        Streamlit dashboard for the trained model
  export_finn.py      Exports the quantized backbone to ONNX for FINN
  src/                Dataset loader, models, training scripts
  checkpoints/        Trained weights and exported ONNX model
```

## Dataset

[ICBHI 2017 Respiratory Sound Database](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database)

## Running the software

```
pip install -r requirements.txt
streamlit run software/app.py
```

Quantization-aware training and the FINN export also need `pip install brevitas`.
