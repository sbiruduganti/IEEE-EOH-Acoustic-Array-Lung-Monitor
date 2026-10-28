# KiCAD files

All hardware design files for the 8-channel ADC acquisition board live here.

## How to open it
1. Install KiCad 7 or newer. Everyone on the team should use the same major version.
2. Open `adc_board.kicad_pro`.
3. Double-click the "ADC" box on the root page to open the ADC sheet.
4. Just want to look? Open `adc_board-schematic.pdf`.

## What's in here
- `adc_board.kicad_pro` / `adc_board.kicad_sch`: the project and its root (top) sheet, which only links the sub-sheets.
- `adc.kicad_sch`: the ADC sheet (2x TLV320ADC5140, 2x THS4524 quad front-end amps, input connector, FPGA header).
- `eoh_adc.kicad_sym` + `sym-lib-table`: the project-local symbol library. Put custom symbols here, not in your personal library.
- `adc-sheet-tables.md`: the connection tables the ADC sheet was built from.
- `gen/`: the script that generated the ADC sheet and the netlist check. If you hand-edit the sheet in KiCad, don't re-run the script.

## Who owns what
- Tommy: `adc.kicad_sch`
- Vimal: `power.kicad_sch` and `test.kicad_sch` (USB-C input, LDOs, PWR_GOOD supervisor, test points), to be added to the root sheet
- Only one person edits the PCB (`adc_board.kicad_pcb`) at a time. Say so in the team chat before you start.

## Rules
- Pull before you start, commit and push when you stop.
- Never edit someone else's sheet without asking. Git can't merge two people's edits to the same KiCad file.
