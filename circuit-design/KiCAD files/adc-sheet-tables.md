# ADC sheet: connection tables

Written 2026-10-26 for Tommy. These tables are what `adc.kicad_sch` is built from. The KiCad netlist was exported and checked against them: all 578 pin connections match.

## How to open it

1. Copy the whole `kicad/` folder to your computer.
2. In KiCad (version 7 or newer), open `adc_board.kicad_pro`.
3. Double-click the "ADC" box on the root page to open the ADC sheet.
4. The project-local symbol library `eoh_adc` (TLV320ADC5140 and ESD2CAN24) is already listed in `sym-lib-table`.
5. If you just want to look at it, open `adc_board-schematic.pdf` instead.

Notes on the file:
- It is drawn in "net label" style. Every pin has a short wire ending in a label, and pins with the same label are connected. That made it possible to check every connection automatically. You can redraw parts with real wires later if you like; just keep the net names.
- Rails (AVDD_ADC, AVDD_FE, IOVDD), PWR_GOOD and GND are global labels, so they join up with Vimal's power sheet once he adds it to the same root page.
- The root page has temporary PWR_FLAG symbols so KiCad's rules check won't complain before the power sheet exists. Delete them once Vimal's LDOs drive the rails. Keep the one on VOCM.
- Every part has an MPN field with the manufacturer part number from bom/EOH-BOM.xlsx, and all 0.1 µF caps are 0603 to match the BOM.
- The amplifiers are 2 quad THS4524IDBTR (TSSOP-38, 0.5 mm pitch): U4 serves channels 1-4 and U5 channels 5-8. It is the same amplifier as the THS4521, four per package (same TI datasheet). Changed 2026-10-26 at Tommy's request. The THS4524 symbol is in eoh_adc.kicad_sym.
- To change values in bulk, edit `gen/gen_adc_sch.py` and re-run it. Hand edits in KiCad are fine too, but then don't re-run the script.

## 1. TLV320ADC5140 pins (U1 = ADC A, U2 = ADC B)

Pin names and numbers were checked against TI's TLV320ADC5140 pin table (RTW, WQFN-24). Footprint: `Texas_RTW_WQFN-24-1EP_4x4mm_P0.5mm_EP2.7x2.7mm_ThermalVias`.

| Pin | Name | U1 (ADC A, 0x4C) | U2 (ADC B, 0x4D) | Parts on the pin |
|---|---|---|---|---|
| 1 | AVDD | AVDD_ADC | AVDD_ADC | 10 µF 0805 + 100 nF 0603 to GND |
| 2 | AREG | AREG_A | AREG_B | 10 µF + 100 nF to GND. Internal 1.8 V regulator output, so don't drive it. Boot script sets AREG_SELECT = 1 |
| 3 | VREF | VREF_A | VREF_B | 1 µF to GND, ground side straight to pin 4, no via |
| 4 | AVSS | GND | GND | |
| 5 | MICBIAS | MICBIAS_A | MICBIAS_B | 1 µF to GND (unused for line input, kept for future sensors) |
| 6 / 7 | IN1P / IN1M | CH1_P / CH1_M | CH5_P / CH5_M | from the front end |
| 8 / 9 | IN2P / IN2M | CH2_P / CH2_M | CH6_P / CH6_M | |
| 10 / 11 | IN3P / IN3M | CH3_P / CH3_M | CH7_P / CH7_M | |
| 12 / 13 | IN4P / IN4M | CH4_P / CH4_M | CH8_P / CH8_M | |
| 14 | SHDNZ | SHDNZ | SHDNZ | R3 100k pull-down to GND (one for both) |
| 15 | ADDR1_MISO | GND | GND | address strap |
| 16 | ADDR0_SCLK | GND | IOVDD | address strap: this is what makes U1 0x4C and U2 0x4D |
| 17 | SCL_MOSI | I2C_SCL | I2C_SCL | R1 2.2k pull-up to IOVDD |
| 18 | SDA_SSZ | I2C_SDA | I2C_SDA | R2 2.2k pull-up to IOVDD |
| 19 | IOVDD | IOVDD (1.8 V) | IOVDD | 1 µF + 100 nF to GND |
| 20 | GPIO1 | IRQ_A_N | IRQ_B_N | straight to MC6. Push-pull, so never join the two |
| 21 | SDOUT | via R4 33R to SDOUT | via R5 33R to SDOUT_B | R6 0R joins SDOUT_B to SDOUT (see below) |
| 22 | BCLK | BCLK | BCLK | from the FPGA via R7 33R at the socket |
| 23 | FSYNC | FSYNC | FSYNC | from the FPGA via R8 33R at the socket |
| 24 | DREG | DREG_A | DREG_B | 10 µF + 100 nF to GND |
| EP | VSS (thermal pad) | GND | GND | via array to the ground plane |

SDOUT link: with R6 fitted, both chips share one data wire (TI's two-chip scheme, which the FPGA boot script expects). With R6 removed, ADC B uses its own wire SDOUT_B on MC6-30, so the two chips can never fight over the line.

Datasheet facts I checked today (TI TLV320ADC5140 datasheet, power-supply section):
- AVDD and IOVDD can come up in either order. Keep SHDNZ low until IOVDD is stable.
- At least 2 ms after SHDNZ goes high before the first I2C write.
- Supply ramps slower than 1 V/µs.
- With AVDD at 3.3 V, the internal AREG regulator is used (AREG_SELECT = 1). Only short AREG to AVDD if AVDD were under 1.98 V.

Still to check: the exact capacitor values on AREG, DREG and IOVDD. They are only in the datasheet's Figure 172 (an image I couldn't read). Open the datasheet PDF, look at that figure, and change the values in the schematic if they differ. The values above come from adc-board-circuit.md §5.

## 2. One front-end channel (the same for all 8)

Reference designators use the channel number in the hundreds: channel 1 is R101–R110, C101–C109, D101, U101; channel 8 is R801–R810, and so on.

| Ref | Value | From | To | Job |
|---|---|---|---|---|
| C_01 | 100 pF C0G | INn_HOT | GND | RF shunt at the connector |
| C_02 | 100 pF C0G | INn_COLD | GND | RF shunt |
| D_01 | ESD2CAN24-Q1 | pin 1 INn_HOT, pin 2 INn_COLD | pin 3 GND | 24 V ESD protection |
| R_01 | 20.0k 0.1% | INn_HOT | FEn_AP | input resistor, sets gain with R_03 |
| R_02 | 20.0k 0.1% | INn_COLD | FEn_AM | |
| C_03 | 390 pF C0G | FEn_AP | FEn_AM | filter cap between the two legs |
| R_03 | 4.02k 0.1% | FEn_AP | FEn_OUTM | feedback (gain = 4.02k / 20.0k = 0.2) |
| R_04 | 4.02k 0.1% | FEn_AM | FEn_OUTP | |
| R_05 | 13.7k | FEn_AP | FEn_INP | to the amplifier + input |
| R_06 | 13.7k | FEn_AM | FEn_INM | to the amplifier − input |
| C_04 | 220 pF C0G | FEn_INP | FEn_OUTM | filter feedback cap |
| C_05 | 220 pF C0G | FEn_INM | FEn_OUTP | |
| U4/U5 section k | THS4524IDBTR (k = channel 1-4 on U4, 5-8 on U5 as k = 1-4) | VINk+ FEn_INP, VINk− FEn_INM, VOCMk VOCM, VSk+ AVDD_FE, PDk AVDD_FE (always on) | VOUTk+ FEn_OUTP, VOUTk− FEn_OUTM | the amplifier. Pins: section k has PD = 5k−4, VIN+ = 5k−3, VIN− = 5k−2, VOCM = 5k−1; VS+/VOUT+/VOUT− = 35/36/37 (k=1), 30/31/32, 25/26/27, 20/21/22 (k=4). VS− pins 5,10,15,23,24,28,29,33,34,38 all GND |
| R_07 | 47R | FEn_OUTP | FEn_RP | isolates the amp from the cap |
| R_08 | 47R | FEn_OUTM | FEn_RM | |
| C_06 | 2.2 nF C0G 0603 | FEn_RP | FEn_RM | charge reservoir for the ADC input |
| R_09 | 0R 1206 | FEn_RP | CHn_P | DC coupling. For the AC fallback, fit 22 µF here |
| R_10 | 0R 1206 | FEn_RM | CHn_M | same |
| C_07 | 100 nF | VOCM | GND | at that section's VOCMk pin |
| C_08 | 100 nF | AVDD_FE | GND | at that section's VSk+ pin |
| C_09 | 1 µF | AVDD_FE | GND | one per pair of channels (odd channels only) |

Sanity check on the polarity: the hot leg feeds the + input, and its feedback comes from OUT−. That makes OUT+ follow HOT, so OUT+ goes to the ADC's P input.

VOCM buffer (one for the board): R9 18.2k from AVDD_ADC and R10 14.7k to GND make 1.474 V (VOCM_DIV), filtered by C21 1 µF. U3 TLV9061 is wired as a follower (+ in = VOCM_DIV, − in = output). Its output goes through R11 100R onto the VOCM net. U3 runs from AVDD_FE with C22 100 nF. Leave R11 off to let each amplifier VOCM float to its own 1.65 V instead.

## 3. Input connector (J1, DB25 female, TASCAM pinout)

This assumes DB25 plus a DB25-to-8×TRS snake. That choice is still open on the card in the thread.

| Channel | Hot | Cold | Ground |
|---|---|---|---|
| 1 | 24 | 12 | 25 |
| 2 | 10 | 23 | 11 |
| 3 | 21 | 9 | 22 |
| 4 | 7 | 20 | 8 |
| 5 | 18 | 6 | 19 |
| 6 | 4 | 17 | 5 |
| 7 | 15 | 3 | 16 |
| 8 | 1 | 14 | 2 |

- Pin 13: not connected.
- The shell goes to GND through R12 0R. If hum shows up, remove R12 and fit R13 100R and C23 10 nF (both DNP for now).
- On the snake, TRS tip = hot, ring = cold, sleeve = ground. Plug channel n of the snake into UMC1820 line output n.
- If we switch to 8 × 1/4 inch jacks, only J1 changes. Each jack's tip goes to INn_HOT, its ring to INn_COLD and its sleeve to GND.

## 4. FPGA plug (J2 onto BRK8305 MC6, J3 onto MC9)

- J2: 2×20 female socket, 2.00 mm pitch (`PinSocket_2x20_P2.00mm_Vertical`).
- J3: 1×20 female socket, 2.00 mm, all 20 pins GND. MC6 has no ground pins, so this is the only ground path to the FPGA.
- All signals are on FPGA bank 86 at 1.8 V. They sit on every other pin, with unused pins as spacers.
- Leave unused MC6 pins open. They are FPGA I/O, so never tie them to GND.

| MC6 pin | FPGA bank net | Our signal | Direction |
|---|---|---|---|
| 1 | B86_L9N | SHDNZ | FPGA → ADCs |
| 5 | B86_L10N | IRQ_A_N | ADC A → FPGA |
| 9 | B86_L11N | IRQ_B_N | ADC B → FPGA |
| 10 | B86_L1P | I2C_SDA | both ways |
| 13 | B86_L12N | PWR_GOOD | our board → FPGA (from Vimal's supervisor) |
| 14 | B86_L2P | I2C_SCL | FPGA → ADCs |
| 18 | B86_L7P_HDGC (ball E13) | BCLK (through R7 33R) | FPGA → ADCs, 12.288 MHz |
| 22 | B86_L8P_HDGC | FSYNC (through R8 33R) | FPGA → ADCs, 48 kHz |
| 26 | B86_L3P | SDOUT | ADCs → FPGA |
| 30 | B86_L4P | SDOUT_B | ADC B → FPGA (only used if R6 is removed) |
| 2, 4, 6 | not connected on the BRK8305 | — | |
| all others | spare | open | |

The pin numbers come from the BRK8305 Rev A03 schematic: odd pins are one row and even pins the other, counted from the end where pins 2, 4 and 6 are unconnected. Send this table to the FPGA team so they can fill in `adc_if.xdc`. Pin 18's ball is E13 (checked earlier); they look up the rest at pins.opalkelly.com.

## 5. Things to check on the real hardware or in documents

1. On the real BRK8305, find pin 1 of MC6 (square pad) and check which row (odd or even) is nearer MC9. I put the clock and data on the even row, assuming it is the one next to the ground row. If it's the other way round, it still works at 12 MHz on a direct plug, but tell me and I'll swap the rows.
2. Check whether MC6 and MC9 already have male headers fitted. If not, solder a 2×20 and a 1×20 2.00 mm male header onto the BRK8305.
3. Read datasheet Figure 172 for the AREG, DREG and IOVDD capacitor values (§1).
4. Confirm the DB25 vs jacks choice (§3).
5. Not drawn on this sheet, add later if wanted: the per-channel sensor-header jumpers (circuit file §10) and the channel-1 test header (Vimal's test-point sheet).
