# Circuit Design

Hardware for the 8-Channel Acoustic Array Lung Monitor, our IEEE UIUC project for Engineering Open House (EOH) 2027.

## The project
The monitor records lung sounds on 8 channels at the same time. An FPGA (Opal Kelly XEM8305) runs a small neural network on those sounds to flag signs of lung problems, such as wheezes and crackles.

## What the circuit design team does
We design and build the ADC board that sits between the audio inputs and the FPGA:
- Analog front end: cleans up and scales each of the 8 input signals
- ADC: turns all 8 channels into digital audio at once (about 16-bit, 48 kHz) and sends it to the FPGA
- Power: quiet, clean supplies for the analog parts
- PCB layout, assembly and hardware testing

## Folder layout
- `KiCAD/`: schematic and PCB files
- `BOM-cost.md`: parts list and cost
- `design-choices.md`: why we chose each part and design
