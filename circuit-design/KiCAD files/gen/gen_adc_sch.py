#!/usr/bin/env python3
"""Generate the ADC sheet (adc.kicad_sch) of the EOH ADC board for KiCad 7+.

Netlabel style: every pin gets a short wire stub ending in a net label, so the
connections are exact and easy to check against the tables in adc-sheet-tables.md.
Re-run this after changing the tables below; it rewrites the .kicad_sch files.

Usage: python3 gen_adc_sch.py <output_dir>
"""
import os, sys, uuid, re

KLIB = "/usr/share/kicad/symbols"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
PROJECT = "adc_board"
ROOT_UUID = "6f1c0a52-7d0e-4f3b-9a51-2b8d3c4e0001"
SHEET_UUID = "6f1c0a52-7d0e-4f3b-9a51-2b8d3c4e0002"

def U():
    return str(uuid.uuid4())

def fmt(v):
    v = round(v, 4)
    return ("%.4f" % v).rstrip("0").rstrip(".") if v != int(v) else str(int(v))

# ---------------------------------------------------------------- symbols
def get_sym(lib, name):
    t = open(os.path.join(KLIB, lib + ".kicad_sym")).read()
    i = t.index('(symbol "%s"' % name)
    d = 0
    for j in range(i, len(t)):
        if t[j] == "(":
            d += 1
        elif t[j] == ")":
            d -= 1
            if d == 0:
                return t[i:j + 1]

def lib_symbol(lib, name):
    """Return (lib_id, embedded text) with 'extends' resolved."""
    s = get_sym(lib, name)
    m = re.match(r'\(symbol "[^"]+" \(extends "([^"]+)"\)', s)
    if m:
        parent = get_sym(lib, m.group(1))
        child_props = re.findall(r'\(property .*?\n\s*\)', s, re.S)
        # take parent's body, swap properties for the child's
        body = re.sub(r'\(property .*?\n\s*\)\n\s*', '', parent, flags=re.S)
        head, rest = body.split("\n", 1)
        head = head.replace('"%s"' % m.group(1), '"%s"' % name)
        rest = rest.replace('(symbol "%s_' % m.group(1), '(symbol "%s_' % name)
        s = head + "\n    " + "\n    ".join(child_props) + "\n" + rest
    lib_id = lib + ":" + name
    s = s.replace('(symbol "%s"' % name, '(symbol "%s"' % lib_id, 1)
    return lib_id, s

def pins_of(text):
    """pin number -> (x, y, angle) in symbol coordinates (y up)."""
    out = {}
    for m in re.finditer(r'\(pin \w+ \w+ \(at ([-\d.]+) ([-\d.]+) (\d+)\).*?\(number "([^"]+)"', text, re.S):
        out[m.group(4)] = (float(m.group(1)), float(m.group(2)), int(m.group(3)))
    return out

# Custom symbols (also written to eoh_adc.kicad_sym)
def box_symbol(name, ref, left, right, bottom, top, w, desc, fp, ds, pin_len=2.54):
    """left/right/top/bottom: lists of (number, name, etype) or None for a gap."""
    n = max(len(left), len(right))
    h = (n + 1) * 2.54
    y0 = (n - 1) * 2.54 / 2
    hb = w / 2
    pins = []
    def add(num, pname, et, x, y, ang):
        pins.append('      (pin %s line (at %s %s %d) (length %s)\n        (name "%s" (effects (font (size 1.27 1.27))))\n        (number "%s" (effects (font (size 1.27 1.27))))\n      )'
                    % (et, fmt(x), fmt(y), ang, fmt(pin_len), pname, num))
    for i, p in enumerate(left):
        if p: add(p[0], p[1], p[2], -hb - pin_len, y0 - i * 2.54, 0)
    for i, p in enumerate(right):
        if p: add(p[0], p[1], p[2], hb + pin_len, y0 - i * 2.54, 180)
    ybot = -(h / 2) - 1.27
    ytop = (h / 2) + 1.27
    for i, p in enumerate(bottom):
        if p: add(p[0], p[1], p[2], (i - (len(bottom) - 1) / 2) * 5.08, ybot - pin_len + 1.27 - 1.27, 90)
    for i, p in enumerate(top):
        if p: add(p[0], p[1], p[2], (i - (len(top) - 1) / 2) * 5.08, ytop + pin_len - 1.27 + 1.27, 270)
    ry = h / 2 + (1.27 if bottom or top else 0)
    ry_top = ytop if top else h / 2
    ry_bot = ybot if bottom else -h / 2
    txt = '''(symbol "%s" (in_bom yes) (on_board yes)
    (property "Reference" "%s" (at 0 %s 0)
      (effects (font (size 1.27 1.27)))
    )
    (property "Value" "%s" (at 0 %s 0)
      (effects (font (size 1.27 1.27)))
    )
    (property "Footprint" "%s" (at 0 0 0)
      (effects (font (size 1.27 1.27)) hide)
    )
    (property "Datasheet" "%s" (at 0 0 0)
      (effects (font (size 1.27 1.27)) hide)
    )
    (property "ki_description" "%s" (at 0 0 0)
      (effects (font (size 1.27 1.27)) hide)
    )
    (symbol "%s_0_1"
      (rectangle (start %s %s) (end %s %s)
        (stroke (width 0.254) (type default))
        (fill (type background))
      )
    )
    (symbol "%s_1_1"
%s
    )
  )''' % (name, ref, fmt(ry_top + 1.27), name, fmt(ry_bot - 1.27), fp, ds, desc, name,
          fmt(-hb), fmt(ry_top), fmt(hb), fmt(ry_bot), name, "\n".join(pins))
    return txt

ADC_SYM = box_symbol(
    "TLV320ADC5140", "U",
    left=[("1", "AVDD", "power_in"), ("2", "AREG", "power_out"), ("3", "VREF", "passive"),
          ("5", "MICBIAS", "power_out"), None,
          ("6", "IN1P_GPI1", "input"), ("7", "IN1M_GPO1", "input"),
          ("8", "IN2P_GPI2", "input"), ("9", "IN2M_GPO2", "input"),
          ("10", "IN3P_GPI3", "input"), ("11", "IN3M_GPO3", "input"),
          ("12", "IN4P_GPI4", "input"), ("13", "IN4M_GPO4", "input")],
    right=[("19", "IOVDD", "power_in"), ("24", "DREG", "power_out"), None,
           ("14", "SHDNZ", "input"), ("16", "ADDR0_SCLK", "input"), ("15", "ADDR1_MISO", "bidirectional"),
           ("17", "SCL_MOSI", "input"), ("18", "SDA_SSZ", "bidirectional"), None,
           ("22", "BCLK", "bidirectional"), ("23", "FSYNC", "bidirectional"),
           ("21", "SDOUT", "tri_state"), ("20", "GPIO1", "bidirectional")],
    bottom=[("4", "AVSS", "power_in"), ("25", "VSS(EP)", "power_in")],
    top=[], w=25.4,
    desc="Quad-channel 768kHz audio ADC, TDM/I2S, I2C/SPI, WQFN-24 4x4 (RTW)",
    fp="Package_DFN_QFN:Texas_RTW_WQFN-24-1EP_4x4mm_P0.5mm_EP2.7x2.7mm_ThermalVias",
    ds="https://www.ti.com/lit/ds/symlink/tlv320adc5140.pdf")

TVS_SYM = box_symbol(
    "ESD2CAN24", "D",
    left=[("1", "IO1", "passive"), ("2", "IO2", "passive")], right=[],
    bottom=[("3", "GND", "passive")], top=[], w=7.62,
    desc="Dual bidirectional 24V TVS, 3pF, SOT-23 (TI ESD2CAN24-Q1)",
    fp="Package_TO_SOT_SMD:SOT-23", ds="https://www.ti.com/lit/ds/symlink/esd2can24-q1.pdf")

_ql, _qr = [], []
for _k in range(1, 5):
    _ql += [("%d" % (5 * _k - 4), "PD%d" % _k, "input"), ("%d" % (5 * _k - 3), "VIN%d+" % _k, "input"),
            ("%d" % (5 * _k - 2), "VIN%d-" % _k, "input"), ("%d" % (5 * _k - 1), "VOCM%d" % _k, "input"), None]
for _k, _vs in zip(range(1, 5), (35, 30, 25, 20)):
    _qr += [("%d" % _vs, "VS%d+" % _k, "power_in"), ("%d" % (_vs + 1), "VOUT%d+" % _k, "output"),
            ("%d" % (_vs + 2), "VOUT%d-" % _k, "output"), None]
_qr += [("%d" % _n, "VS-", "power_in") for _n in (5, 10, 15, 23, 24, 28, 29, 33, 34, 38)]
QUAD_SYM = box_symbol(
    "THS4524", "U", left=_ql[:-1], right=_qr, bottom=[], top=[], w=17.78,
    desc="Quad very-low-power fully differential amplifier, RRO, negative-rail input, TSSOP-38 (DBT, 0.5 mm pitch)",
    fp="Package_SO:TSSOP-38_4.4x9.7mm_P0.5mm", ds="https://www.ti.com/lit/ds/symlink/ths4524.pdf")
# all VS- pins are the same net; KiCad would flag stacked power pins otherwise
CUSTOM = {"TLV320ADC5140": ADC_SYM, "ESD2CAN24": TVS_SYM, "THS4524": QUAD_SYM}

SYMS = {}  # lib_id -> (embedded text, pins)
def need(lib, name):
    if lib == "eoh_adc":
        lib_id = "eoh_adc:" + name
        if lib_id not in SYMS:
            t = CUSTOM[name].replace('(symbol "%s"' % name, '(symbol "%s"' % lib_id, 1)
            SYMS[lib_id] = (t, pins_of(t))
        return lib_id
    lib_id, t = lib_symbol(lib, name)
    if lib_id not in SYMS:
        SYMS[lib_id] = (t, pins_of(t))
    return lib_id

# ---------------------------------------------------------------- drawing
ITEMS = []
INTENDED = {}   # (ref, pin) -> net, for the self-check
REFS = {}

GLOBAL_NETS = set()

def rot(px, py, a):
    # symbol coords (y up) -> screen offset (y down), rotated CCW by a degrees
    x, y = px, -py
    if a == 0: return x, y
    if a == 90: return y, -x
    if a == 180: return -x, -y
    if a == 270: return -y, x

def label(net, x, y, ang):
    if net == "GND":
        r = {270: 0, 90: 180, 0: 90, 180: 270}[ang]
        ITEMS.append('(symbol (lib_id "power:GND") (at %s %s %d) (unit 1) (in_bom yes) (on_board yes) (dnp no) (uuid %s)\n'
                     '  (property "Reference" "#PWR?" (at %s %s 0) (effects (font (size 1.27 1.27)) hide))\n'
                     '  (property "Value" "GND" (at %s %s 0) (effects (font (size 1.27 1.27)) hide))\n'
                     '  (property "Footprint" "" (at %s %s 0) (effects (font (size 1.27 1.27)) hide))\n'
                     '  (property "Datasheet" "" (at %s %s 0) (effects (font (size 1.27 1.27)) hide))\n'
                     '  (pin "1" (uuid %s))\n  (instances (project "%s" (path "/%s/%s" (reference "#PWR0%d") (unit 1))))\n)'
                     % (fmt(x), fmt(y), r, U(), fmt(x), fmt(y), fmt(x), fmt(y), fmt(x), fmt(y), fmt(x), fmt(y), U(),
                        PROJECT, ROOT_UUID, SHEET_UUID, next_pwr()))
        return
    just = "left" if ang in (0, 90) else "right"
    if net in GLOBAL_NETS:
        ITEMS.append('(global_label "%s" (shape passive) (at %s %s %d) (fields_autoplaced)\n'
                     '  (effects (font (size 1.27 1.27)) (justify %s)) (uuid %s)\n'
                     '  (property "Intersheetrefs" "${INTERSHEET_REFS}" (at %s %s 0) (effects (font (size 1.27 1.27)) (justify %s) hide))\n)'
                     % (net, fmt(x), fmt(y), ang, just, U(), fmt(x), fmt(y), just))
    else:
        ITEMS.append('(label "%s" (at %s %s %d) (fields_autoplaced)\n  (effects (font (size 1.27 1.27)) (justify %s bottom)) (uuid %s)\n)'
                     % (net, fmt(x), fmt(y), ang, just, U()))

_pwr = [0]
def next_pwr():
    _pwr[0] += 1
    return _pwr[0]

def wire(x1, y1, x2, y2):
    ITEMS.append('(wire (pts (xy %s %s) (xy %s %s)) (stroke (width 0) (type default)) (uuid %s))'
                 % (fmt(x1), fmt(y1), fmt(x2), fmt(y2), U()))

def text(s, x, y, size=1.27, bold=False):
    s = s.replace('"', "'")
    ITEMS.append('(text "%s" (at %s %s 0)\n  (effects (font (size %s %s)%s) (justify left bottom)) (uuid %s)\n)'
                 % (s, fmt(x), fmt(y), fmt(size), fmt(size), " bold" if bold else "", U()))

def place(lib, name, ref, value, x, y, conns, a=0, fp=None, dnp=False, stub=2.54, extra=None):
    """conns: pin number -> net name, or None for no-connect."""
    lib_id = need(lib, name)
    text_, pins = SYMS[lib_id]
    assert ref not in REFS, ref
    REFS[ref] = value
    if fp is None:
        m = re.search(r'\(property "Footprint" "([^"]*)"', text_)
        fp = m.group(1) if m else ""
    props = [("Reference", ref), ("Value", value), ("Footprint", fp), ("Datasheet", "")]
    if extra:
        props += list(extra.items())
    elif value in MPN:
        props.append(("MPN", MPN[value]))
    # property positions: next to the body
    bx, by = rot(5.08, 0, a)
    s = '(symbol (lib_id "%s") (at %s %s %d) (unit 1) (in_bom yes) (on_board yes) (dnp %s) (uuid %s)\n' % (
        lib_id, fmt(x), fmt(y), a, "yes" if dnp else "no", U())
    vert = a in (0, 180) and len(pins) == 2
    for i, (k, v) in enumerate(props):
        if vert:
            px, py, just = x + 2.54, y - 1.27 + i * 2.54, "left"
        elif len(pins) == 2:
            px, py, just = x, y - 3.81 - (1 - i) * 2.54, "left"
            px -= 2.54 if i < 2 else 0
            just = "left"
        else:
            px, py, just = x, y + (i * 2.54) + 0, "left"
        hide = " hide" if i >= 2 else ""
        if len(pins) > 2 and i < 2:
            # big parts: put ref/value above the body
            ymin = min(-p[1] for p in pins.values())
            xmax = max(p[0] for p in pins.values())
            px, py = x + xmax + 2.54, y + ymin - 2.54 + i * 2.54
            if not any(p[2] == 270 for p in pins.values()):
                px, py = x - 5.08, y + ymin - 6.35 + i * 2.54
        s += '  (property "%s" "%s" (at %s %s 0) (effects (font (size 1.27 1.27)) (justify %s)%s))\n' % (
            k, v, fmt(px), fmt(py), just, hide)
    for num in pins:
        s += '  (pin "%s" (uuid %s))\n' % (num, U())
    s += '  (instances (project "%s" (path "/%s/%s" (reference "%s") (unit 1))))\n)' % (
        PROJECT, ROOT_UUID, SHEET_UUID, ref)
    ITEMS.append(s)
    for num, (px, py, pa) in pins.items():
        if num not in conns:
            raise SystemExit("unassigned pin %s.%s" % (ref, num))
        dx, dy = rot(px, py, a)
        tx, ty = x + dx, y + dy
        net = conns[num]
        if net is None:
            ITEMS.append('(no_connect (at %s %s) (uuid %s))' % (fmt(tx), fmt(ty), U()))
            continue
        INTENDED[(ref, num)] = net
        out = (pa + a + 180) % 360
        ox, oy = {0: (1, 0), 90: (0, -1), 180: (-1, 0), 270: (0, 1)}[out]
        ex, ey = tx + ox * stub, ty + oy * stub
        wire(tx, ty, ex, ey)
        label(net, ex, ey, out)
    for k in conns:
        if k not in pins:
            raise SystemExit("bad pin %s.%s" % (ref, k))

# Manufacturer part numbers, matching bom/EOH-BOM.xlsx (2026-10-26)
MPN = {
    "20.0k 0.1%": "Susumu RG1608P-203-B-T5", "4.02k 0.1%": "Susumu RG1608P-4021-B-T5",
    "13.7k": "Yageo RC0603FR-0713K7L", "47R": "Yageo RC0603FR-0747RL",
    "220p C0G": "Murata GRM1885C1H221GA01D", "390p C0G": "Murata GRM1885C1H391GA01D",
    "2.2n C0G": "KEMET C0603C222G5GACTU", "100p C0G": "Murata GRM1885C1H101JA01D",
    "10uF": "Samsung CL21B106KOQNNNE", "1uF": "KEMET C0603C105K4RACTU", "1u": "KEMET C0603C105K4RACTU",
    "100nF": "KEMET C0603C104K5RACTU", "100n": "KEMET C0603C104K5RACTU",
    "2.2k": "Yageo RC0603FR-072K2L", "33R": "Yageo RC0603FR-0733RL", "100k": "Yageo RC0603FR-07100KL",
    "18.2k 1%": "Yageo RC0603FR-0718K2L", "14.7k 1%": "Yageo RC0603FR-0714K7L", "100R": "Yageo RC0603FR-07100RL",
    "0R (link)": "Yageo RC0603JR-070RL",
    "THS4524IDBTR": "TI THS4524IDBTR", "TLV320ADC5140IRTWR": "TI TLV320ADC5140IRTWR",
    "TLV9061IDBVR": "TI TLV9061IDBVR", "ESD2CAN24-Q1": "TI ESD2CAN24DBZRQ1",
    "DB25 female R/A": "Amphenol L77SDB25S1ACH4F",
    "MC6 socket 2x20 2.00mm": "Harwin M22-7142042", "MC9 socket 1x20 2.00mm (GND)": "Harwin M22-7132042",
}

def R(ref, val, x, y, n1, n2, fp="Resistor_SMD:R_0603_1608Metric", a=0, dnp=False):
    mpn = MPN.get(val, "Yageo RC1206JR-070RL" if fp.endswith("1206_3216Metric") else ("Yageo RC0603JR-070RL" if val.startswith("0R") else ""))
    place("Device", "R", ref, val, x, y, {"1": n1, "2": n2}, a=a, fp=fp, dnp=dnp, extra={"MPN": mpn})

def C(ref, val, x, y, n1, n2, fp="Capacitor_SMD:C_0603_1608Metric", a=0, dnp=False):
    place("Device", "C", ref, val, x, y, {"1": n1, "2": n2}, a=a, fp=fp, dnp=dnp, extra={"MPN": MPN.get(val, "")})

# ---------------------------------------------------------------- design data
GLOBAL_NETS |= {"AVDD_ADC", "AVDD_FE", "IOVDD", "PWR_GOOD", "SHDNZ", "I2C_SCL", "I2C_SDA",
                "BCLK", "FSYNC", "SDOUT", "SDOUT_B", "IRQ_A_N", "IRQ_B_N", "VOCM",
                "BCLK_FPGA", "FSYNC_FPGA"}

# DB25, TASCAM/AES59 analog pinout: channel -> (hot, cold, ground)
DB25 = {1: (24, 12, 25), 2: (10, 23, 11), 3: (21, 9, 22), 4: (7, 20, 8),
        5: (18, 6, 19), 6: (4, 17, 5), 7: (15, 3, 16), 8: (1, 14, 2)}

# BRK8305 MC6 (our J2) pin -> (FPGA bank net, our signal). Pins 2/4/6 are NC on the BRK8305.
MC6 = {
    1: ("B86_L9N", "SHDNZ"), 3: ("B86_L9P", None), 5: ("B86_L10N", "IRQ_A_N"), 7: ("B86_L10P", None),
    9: ("B86_L11N", "IRQ_B_N"), 11: ("B86_L11P", None), 13: ("B86_L12N", "PWR_GOOD"), 15: ("B86_L12P", None),
    17: ("B85_L12N", None), 19: ("B85_L12P", None), 21: ("B85_L11N", None), 23: ("B85_L11P", None),
    25: ("B85_L10N", None), 27: ("B85_L10P", None), 29: ("B85_L9N", None), 31: ("B85_L9P", None),
    33: ("B85_L8N_HDGC", None), 35: ("B85_L8P_HDGC", None), 37: ("B85_L7N_HDGC", None), 39: ("B85_L7P_HDGC", None),
    2: ("NC", None), 4: ("NC", None), 6: ("NC", None),
    8: ("B86_L1N", None), 10: ("B86_L1P", "I2C_SDA"), 12: ("B86_L2N", None), 14: ("B86_L2P", "I2C_SCL"),
    16: ("B86_L7N_HDGC", None), 18: ("B86_L7P_HDGC", "BCLK_FPGA"), 20: ("B86_L8N_HDGC", None),
    22: ("B86_L8P_HDGC", "FSYNC_FPGA"), 24: ("B86_L3N", None), 26: ("B86_L3P", "SDOUT"), 28: ("B86_L4N", None),
    30: ("B86_L4P", "SDOUT_B"), 32: ("B86_L5N_HDGC", None), 34: ("B86_L5P_HDGC", None),
    36: ("B86_L6N_HDGC", None), 38: ("B86_L6P_HDGC", None), 40: ("B85_L6N_HDGC", None),
}

FP_0603C = "Capacitor_SMD:C_0603_1608Metric"
FP_0805C = "Capacitor_SMD:C_0805_2012Metric"
FP_0402C = "Capacitor_SMD:C_0603_1608Metric"  # BOM buys 0.1 uF in 0603 (KEMET C0603C104K5RACTU)

def build():
    T = text
    T("ADC SHEET - EOH lung array ADC board (owner: Tommy). 2x TLV320ADC5140 + 2x THS4524 (quad) front end + DB25 input + BRK8305 MC6/MC9 plug", 15, 15, 2.5, True)
    T("Generated from gen_adc_sch.py (netlabel style). Pins connect by the label at the end of each stub. Rails (AVDD_ADC, AVDD_FE, IOVDD), PWR_GOOD and test points come from Vimal's power sheet.", 15, 21, 1.5)
    T("Source of values: project-map/adc-board-circuit.md (front end, decoupling), fpga-interface.md (I2C addresses, TDM, pins). Tables: kicad/adc-sheet-tables.md", 15, 25, 1.5)

    # ---------------- input connector
    x0, y0 = 35, 120
    T("INPUT J1: DB25 female, TASCAM pinout", 8, 70, 1.5, True)
    T("UMC1820 8 TRS line outs -> DB25-to-8xTRS snake", 8, 74, 1.27)
    conns = {"0": "DB25_SHELL", "13": None}
    for ch, (h, c, g) in DB25.items():
        conns[str(h)] = "IN%d_HOT" % ch
        conns[str(c)] = "IN%d_COLD" % ch
        conns[str(g)] = "GND"
    place("Connector", "DB25_Receptacle_MountingHoles", "J1", "DB25 female R/A", x0, y0, conns,
          fp="Connector_Dsub:DSUB-25_Female_Horizontal_P2.77x2.84mm_EdgePinOffset9.90mm_Housed_MountingHolesOffset11.32mm",
          stub=5.08)
    T("Shell ground-lift: R12 fitted 0R; if hum, remove R12 and fit R13+C23", 12, 170, 1.27)
    R("R12", "0R", 20, 185, "DB25_SHELL", "GND")
    R("R13", "100R DNP", 32, 185, "DB25_SHELL", "GND", dnp=True)
    C("C23", "10nF DNP", 44, 185, "DB25_SHELL", "GND", dnp=True)

    # ---------------- 8 front-end channels
    quad_conns = {}
    for ch in range(1, 9):
        col, row = (ch - 1) // 4, (ch - 1) % 4
        ox, oy = 75 + col * 235, 45 + row * 95
        n = ch
        H, Cn = "IN%d_HOT" % n, "IN%d_COLD" % n
        AP, AM = "FE%d_AP" % n, "FE%d_AM" % n
        IP, IM = "FE%d_INP" % n, "FE%d_INM" % n
        OP, OM = "FE%d_OUTP" % n, "FE%d_OUTM" % n
        RP, RM = "FE%d_RP" % n, "FE%d_RM" % n
        AdP, AdM = "CH%d_P" % n, "CH%d_M" % n
        T("CH%d front end: THS4524 diff MFB low-pass, gain 0.2 (-14 dB), 50 kHz Butterworth, DC-coupled to ADC %s IN%d"
          % (n, "A (U1)" if n <= 4 else "B (U2)", (n - 1) % 4 + 1), ox, oy, 1.5, True)
        b = n * 100
        y1 = oy + 22
        xs = [ox + 4 + i * 15.24 for i in range(12)]
        C("C%d" % (b + 1), "100p C0G", xs[0], y1, H, "GND", fp=FP_0603C)
        C("C%d" % (b + 2), "100p C0G", xs[1], y1, Cn, "GND", fp=FP_0603C)
        R("R%d" % (b + 1), "20.0k 0.1%", xs[2], y1, H, AP)
        R("R%d" % (b + 2), "20.0k 0.1%", xs[3], y1, Cn, AM)
        C("C%d" % (b + 3), "390p C0G", xs[4], y1, AP, AM, fp=FP_0603C)
        R("R%d" % (b + 3), "4.02k 0.1%", xs[5], y1, AP, OM)
        R("R%d" % (b + 4), "4.02k 0.1%", xs[6], y1, AM, OP)
        R("R%d" % (b + 5), "13.7k", xs[7], y1, AP, IP)
        R("R%d" % (b + 6), "13.7k", xs[8], y1, AM, IM)
        C("C%d" % (b + 4), "220p C0G", xs[9], y1, IP, OM, fp=FP_0603C)
        C("C%d" % (b + 5), "220p C0G", xs[10], y1, IM, OP, fp=FP_0603C)
        y2 = oy + 62
        R("R%d" % (b + 7), "47R", xs[0], y2, OP, RP)
        R("R%d" % (b + 8), "47R", xs[1], y2, OM, RM)
        C("C%d" % (b + 6), "2.2n C0G", xs[2], y2, RP, RM, fp=FP_0603C)
        R("R%d" % (b + 9), "0R", xs[3], y2, RP, AdP, fp="Resistor_SMD:R_1206_3216Metric")
        R("R%d" % (b + 10), "0R", xs[4], y2, RM, AdM, fp="Resistor_SMD:R_1206_3216Metric")
        C("C%d" % (b + 7), "100n", xs[6], y2, "VOCM", "GND", fp=FP_0402C)
        C("C%d" % (b + 8), "100n", xs[7], y2, "AVDD_FE", "GND", fp=FP_0402C)
        T("R%d/R%d = Cc position (1206): 0R = DC-coupled (default). AC fallback: fit 22uF and set CHx_CFG0 = 0x84" % (b + 9, b + 10), xs[0], y2 + 22, 1.0)
        if n % 2 == 1:
            C("C%d" % (b + 9), "1u", xs[8], y2, "AVDD_FE", "GND", fp=FP_0603C)
        # TVS and FDA
        place("eoh_adc", "ESD2CAN24", "D%d" % (b + 1), "ESD2CAN24-Q1", ox + 192, oy + 18,
              {"1": H, "2": Cn, "3": "GND"})
        k = (n - 1) % 4 + 1
        vs = {1: 35, 2: 30, 3: 25, 4: 20}[k]
        quad_conns.setdefault("U%d" % (4 if n <= 4 else 5), {}).update({
            str(5 * k - 4): "AVDD_FE", str(5 * k - 3): IP, str(5 * k - 2): IM, str(5 * k - 1): "VOCM",
            str(vs): "AVDD_FE", str(vs + 1): OP, str(vs + 2): OM})
        T("Amplifier: %s section %d (VIN%d+/- , VOUT%d+/-), on the quad THS4524 below" % ("U4" if n <= 4 else "U5", k, k, k), ox + 170, oy + 58, 1.27)

    # ---------------- quad amplifiers
    T("FRONT-END AMPLIFIERS: 2x THS4524 quad fully differential amp (U4 = CH1-4, U5 = CH5-8). Same amplifier as THS4521, four per package.", 250, 470, 1.5, True)
    T("Every PD pin tied to AVDD_FE (always on). All VS- pins to GND. Each VSn+ pin has its own 100n (C_08 of that channel), each VOCMn pin its own 100n (C_07).", 250, 474, 1.27)
    for i, (ref, qc) in enumerate(sorted(quad_conns.items())):
        for g in (5, 10, 15, 23, 24, 28, 29, 33, 34, 38):
            qc[str(g)] = "GND"
        place("eoh_adc", "THS4524", ref, "THS4524IDBTR", 290 + i * 120, 515, qc, stub=5.08)

    # ---------------- VOCM buffer
    vx, vy = 75, 440
    T("VOCM: 1.474 V from AVDD_ADC divider, buffered by TLV9061 (matches ADC5140 internal input bias ~1.48 V). DNP R11 to let each THS4524 VOCM float to 1.65 V", vx, vy - 12, 1.5, True)
    R("R9", "18.2k 1%", vx, vy + 5, "AVDD_ADC", "VOCM_DIV")
    R("R10", "14.7k 1%", vx + 11.43, vy + 5, "VOCM_DIV", "GND")
    C("C21", "1uF", vx + 22.86, vy + 5, "VOCM_DIV", "GND", fp=FP_0603C)
    place("Amplifier_Operational", "TLV9061xDBV", "U3", "TLV9061IDBVR", vx + 50, vy + 5,
          {"3": "VOCM_DIV", "4": "VOCM_BUF", "1": "VOCM_BUF", "5": "AVDD_FE", "2": "GND"})
    R("R11", "100R", vx + 75, vy + 5, "VOCM_BUF", "VOCM")
    C("C22", "100nF", vx + 86.43, vy + 5, "AVDD_FE", "GND", fp=FP_0402C)

    # ---------------- ADCs
    for idx, (ref, addr0, slot) in enumerate([("U1", "GND", "A"), ("U2", "IOVDD", "B")]):
        ax, ay = 600, 75 + idx * 150
        chs = range(1, 5) if idx == 0 else range(5, 9)
        T("ADC %s: TLV320ADC5140 (%s), I2C 0x%s, channels %d-%d -> TDM slots %d-%d" % (
            slot, ref, "4C" if idx == 0 else "4D", chs[0], chs[-1], chs[0] - 1, chs[-1] - 1), ax - 40, ay - 35, 1.5, True)
        k = slot
        conns = {"1": "AVDD_ADC", "2": "AREG_" + k, "3": "VREF_" + k, "5": "MICBIAS_" + k,
                 "4": "GND", "25": "GND", "19": "IOVDD", "24": "DREG_" + k,
                 "14": "SHDNZ", "16": addr0, "15": "GND", "17": "I2C_SCL", "18": "I2C_SDA",
                 "22": "BCLK", "23": "FSYNC", "21": "SDOUT_RAW_" + k, "20": "IRQ_%s_N" % k}
        for i, chn in enumerate(chs):
            conns[str(6 + 2 * i)] = "CH%d_P" % chn
            conns[str(7 + 2 * i)] = "CH%d_M" % chn
        place("eoh_adc", "TLV320ADC5140", ref, "TLV320ADC5140IRTWR", ax, ay, conns, stub=5.08)
        cb = idx * 10
        cx, cy = ax + 40, ay - 20
        caps = [("10uF", "AVDD_ADC", FP_0805C), ("100nF", "AVDD_ADC", FP_0402C),
                ("10uF", "AREG_" + k, FP_0805C), ("100nF", "AREG_" + k, FP_0402C),
                ("1uF", "VREF_" + k, FP_0603C), ("1uF", "MICBIAS_" + k, FP_0603C),
                ("1uF", "IOVDD", FP_0603C), ("100nF", "IOVDD", FP_0402C),
                ("10uF", "DREG_" + k, FP_0805C), ("100nF", "DREG_" + k, FP_0402C)]
        for i, (val, net, fp) in enumerate(caps):
            C("C%d" % (cb + i + 1), val, cx + (i % 5) * 13, cy + (i // 5) * 40, net, "GND", fp=fp)
        T("%s decoupling: each cap right at its pin." % ref, cx - 5, cy + 62, 1.27)
        T("VREF/MICBIAS cap grounds go straight to AVSS pin 4, no via.", cx - 5, cy + 65, 1.27)
        T("Confirm values vs datasheet Fig. 172 / EVM schematic.", cx - 5, cy + 68, 1.27)

    # ---------------- digital glue
    gx, gy = 610, 400
    T("DIGITAL GLUE: I2C pull-ups, SHDNZ pull-down, 33R series, SDOUT_B link", gx - 40, gy - 22, 1.5, True)
    R("R1", "2.2k", gx - 30, gy, "IOVDD", "I2C_SCL")
    R("R2", "2.2k", gx - 18, gy, "IOVDD", "I2C_SDA")
    R("R3", "100k", gx - 6, gy, "SHDNZ", "GND")
    R("R4", "33R", gx + 8, gy, "SDOUT_RAW_A", "SDOUT")
    R("R5", "33R", gx + 20, gy, "SDOUT_RAW_B", "SDOUT_B")
    R("R6", "0R (link)", gx + 32, gy, "SDOUT_B", "SDOUT")
    R("R7", "33R", gx + 46, gy, "BCLK_FPGA", "BCLK")
    R("R8", "33R", gx + 58, gy, "FSYNC_FPGA", "FSYNC")
    T("R6 fitted: both ADCs share SDOUT (TI scheme). R6 removed: ADC B uses its own lane SDOUT_B (MC6-30).", gx - 40, gy + 20, 1.27)
    T("R4/R5 sit at each ADC's SDOUT pin; R7/R8 sit at the MC6 socket.", gx - 40, gy + 23, 1.27)

    # ---------------- FPGA plug
    fx, fy = 765, 120
    T("FPGA PLUG J2/J3: 2x20 2.00 mm socket onto BRK8305 MC6", fx - 50, fy - 49, 1.5, True)
    T("MC6 = banks 85/86 at 1.8 V, no ground pins. J3 1x20 socket onto MC9 = all GND (ground return).", fx - 50, fy - 45, 1.27)
    T("All signals on bank 86. Unused MC6 pins are left open (they are FPGA I/O: never tie them to GND).", fx - 50, fy - 41, 1.27)
    T("Check on the real BRK8305 which MC6 row (odd or even) is nearer MC9; the clock pins (BCLK 18, FSYNC 22, SDOUT 26) are on the even row.", fx - 50, fy - 38, 1.27)
    conns = {}
    for p, (bank, sig) in MC6.items():
        conns[str(p)] = sig
    place("Connector_Generic", "Conn_02x20_Odd_Even", "J2", "MC6 socket 2x20 2.00mm", fx, fy, conns,
          fp="Connector_PinSocket_2.00mm:PinSocket_2x20_P2.00mm_Vertical", stub=5.08)
    for p, (bank, sig) in MC6.items():
        _, pins = SYMS["Connector_Generic:Conn_02x20_Odd_Even"]
        px, py, pa = pins[str(p)]
        tx = fx - 48 if pa == 0 else fx + 36
        T("MC6-%d = %s" % (p, bank), tx, fy - py + 0.5, 1.0)
    place("Connector_Generic", "Conn_01x20", "J3", "MC9 socket 1x20 2.00mm (GND)", fx, fy + 80,
          {str(i): "GND" for i in range(1, 21)}, fp="Connector_PinSocket_2.00mm:PinSocket_1x20_P2.00mm_Vertical")

def lib_symbols_block():
    need("power", "GND")
    return "(lib_symbols\n" + "\n".join(SYMS[k][0] for k in sorted(SYMS)) + "\n)"

def write():
    build()
    ls = lib_symbols_block()
    with open(os.path.join(OUT, "adc.kicad_sch"), "w") as f:
        f.write('(kicad_sch (version 20230121) (generator eeschema)\n(uuid %s)\n(paper "A1")\n' % SHEET_UUID)
        f.write('(title_block (title "ADC board - ADC sheet") (date "2026-10-26") (rev "0.1") (company "IEEE UIUC EOH 2027") (comment 1 "Owner: Tommy. Generated by gen_adc_sch.py"))\n')
        f.write(ls + "\n")
        f.write("\n".join(ITEMS))
        f.write("\n)\n")
    # root sheet with the ADC sheet + temporary power flags
    flags = []
    pf = need("power", "PWR_FLAG")
    for i, net in enumerate(["AVDD_ADC", "AVDD_FE", "IOVDD", "GND", "VOCM"]):
        x, y = 40 + i * 30, 150
        flags.append('(symbol (lib_id "power:PWR_FLAG") (at %s %s 0) (unit 1) (in_bom yes) (on_board yes) (dnp no) (uuid %s)\n'
                     '  (property "Reference" "#FLG0%d" (at %s %s 0) (effects (font (size 1.27 1.27)) hide))\n'
                     '  (property "Value" "PWR_FLAG" (at %s %s 0) (effects (font (size 1.27 1.27))))\n'
                     '  (property "Footprint" "" (at %s %s 0) (effects (font (size 1.27 1.27)) hide))\n'
                     '  (property "Datasheet" "~" (at %s %s 0) (effects (font (size 1.27 1.27)) hide))\n'
                     '  (pin "1" (uuid %s))\n  (instances (project "%s" (path "/%s" (reference "#FLG0%d") (unit 1))))\n)'
                     % (x, y, U(), i + 1, x, y - 2, x, y - 4, x, y, x, y, U(), PROJECT, ROOT_UUID, i + 1))
        flags.append('(wire (pts (xy %s %s) (xy %s %s)) (stroke (width 0) (type default)) (uuid %s))' % (x, y, x, y + 5.08, U()))
        if net == "GND":
            flags.append('(symbol (lib_id "power:GND") (at %s %s 0) (unit 1) (in_bom yes) (on_board yes) (dnp no) (uuid %s)\n'
                         '  (property "Reference" "#PWR0900" (at %s %s 0) (effects (font (size 1.27 1.27)) hide))\n'
                         '  (property "Value" "GND" (at %s %s 0) (effects (font (size 1.27 1.27))))\n'
                         '  (property "Footprint" "" (at %s %s 0) (effects (font (size 1.27 1.27)) hide))\n'
                         '  (property "Datasheet" "" (at %s %s 0) (effects (font (size 1.27 1.27)) hide))\n'
                         '  (pin "1" (uuid %s))\n  (instances (project "%s" (path "/%s" (reference "#PWR0900") (unit 1))))\n)'
                         % (x, y + 5.08, U(), x, y + 9, x, y + 9, x, y + 9, x, y + 9, U(), PROJECT, ROOT_UUID))
        else:
            flags.append('(global_label "%s" (shape passive) (at %s %s 270) (fields_autoplaced)\n  (effects (font (size 1.27 1.27)) (justify right)) (uuid %s)\n'
                         '  (property "Intersheetrefs" "${INTERSHEET_REFS}" (at %s %s 0) (effects (font (size 1.27 1.27)) (justify right) hide))\n)'
                         % (net, x, y + 5.08, U(), x, y + 5.08))
    root_syms = "(lib_symbols\n" + SYMS["power:PWR_FLAG"][0] + "\n" + SYMS["power:GND"][0] + "\n)"
    with open(os.path.join(OUT, PROJECT + ".kicad_sch"), "w") as f:
        f.write('(kicad_sch (version 20230121) (generator eeschema)\n(uuid %s)\n(paper "A4")\n' % ROOT_UUID)
        f.write('(title_block (title "EOH ADC board - root") (date "2026-10-26") (rev "0.1") (company "IEEE UIUC EOH 2027"))\n')
        f.write(root_syms + "\n")
        f.write('(text "Root sheet. Vimal adds power.kicad_sch and test.kicad_sch here as more sheets. The PWR_FLAGs below are TEMPORARY so ERC passes before the power sheet exists: delete them once the LDO outputs drive AVDD_ADC, AVDD_FE and IOVDD (VOCM is driven by U3 on the ADC sheet but needs a flag because a 100R sits between)." (at 20 120 0)\n  (effects (font (size 1.5 1.5)) (justify left bottom)) (uuid %s)\n)\n' % U())
        f.write('(sheet (at 40 40) (size 60 40) (fields_autoplaced)\n  (stroke (width 0.1524) (type solid)) (fill (color 0 0 0 0.0000)) (uuid %s)\n'
                '  (property "Sheetname" "ADC" (at 40 39.3 0) (effects (font (size 1.27 1.27)) (justify left bottom)))\n'
                '  (property "Sheetfile" "adc.kicad_sch" (at 40 80.6 0) (effects (font (size 1.27 1.27)) (justify left top)))\n'
                '  (instances (project "%s" (path "/%s" (page "2"))))\n)\n' % (SHEET_UUID, PROJECT, ROOT_UUID))
        f.write("\n".join(flags))
        f.write('\n(sheet_instances (path "/" (page "1")))\n)\n')
    # project-local symbol library
    with open(os.path.join(OUT, "eoh_adc.kicad_sym"), "w") as f:
        f.write("(kicad_symbol_lib (version 20220914) (generator gen_adc_sch)\n  " + ADC_SYM + "\n  " + TVS_SYM + "\n)\n")
    with open(os.path.join(OUT, "sym-lib-table"), "w") as f:
        f.write('(sym_lib_table\n  (lib (name "eoh_adc")(type "KiCad")(uri "${KIPRJMOD}/eoh_adc.kicad_sym")(options "")(descr "EOH ADC board parts"))\n)\n')
    import json
    with open(os.path.join(OUT, "intended_nets.json"), "w") as f:
        json.dump({"%s.%s" % k: v for k, v in INTENDED.items()}, f, indent=0, sort_keys=True)
    if not os.path.exists(os.path.join(OUT, PROJECT + ".kicad_pro")):
        with open(os.path.join(OUT, PROJECT + ".kicad_pro"), "w") as f:
            json.dump({"meta": {"filename": PROJECT + ".kicad_pro", "version": 1}, "schematic": {}, "boards": [], "sheets": [[ROOT_UUID, ""], [SHEET_UUID, "ADC"]]}, f, indent=2)
    print("parts:", len(REFS), "pins:", len(INTENDED))

write()
