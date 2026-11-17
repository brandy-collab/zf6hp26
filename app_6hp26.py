"""
Simulador de la caja automática ZF 6HP26 (disposición Lepelletier).

Esquema como en la figura del artículo: convertidor de par, tren simple (fila 1)
con el planeta fijo, embragues C1234, C35R, C456, frenos B1R, B26 y tren
Ravigneaux (filas 2 y 3) con corona de salida.

Notación (la de la clase «Relaciones de cambio de la 6HP26»):
    s planeta, r corona, p satélite, c portasatélites
    fila 1: z_s1, z_r1, z_p1           (ω_r1 = ω_ent, ω_s1 = 0)
    fila 2: z_s2, z_r2, z_p2           (planeta grande, satélite largo)
    fila 3: z_s3, z_r3, z_p3b, z_p3d   (planeta pequeño, satélites largo y corto)
    ω_c2 = ω_c3 = ω_c,   ω_r2 = ω_r3 = ω_sal

Ecuaciones de Willis:
    (1) ω_r1 - ω_c1 = -(z_s1/z_r1) (ω_s1 - ω_c1)
    (2) ω_sal - ω_c = -(z_s2/z_r2) (ω_s2 - ω_c)
    (3) ω_sal - ω_c = +(z_s3/z_r3) (ω_s3 - ω_c)

Ejecutar con:   streamlit run app_6hp26.py
"""

import json
import math

import numpy as np
import streamlit as st

st.set_page_config(page_title="Caja ZF 6HP26", layout="wide")

# ---------------------------------------------------------------------------
#  Datos
# ---------------------------------------------------------------------------
Z0 = dict(s1=37, r1=71, p1=17, s2=38, r2=85, p2=23, s3=31, r3=85, p3d=28)

ELEMENTOS = ["C1234", "C456", "C35R", "B1R", "B26"]
NOMBRE_EL = {"C1234": "C₁,₂,₃,₄", "C456": "C₄,₅,₆", "C35R": "C₃,₅,R", "B1R": "B₁,R", "B26": "B₂,₆"}
UNE = {"C1234": "c₁ ↔ s₃", "C456": "entrada ↔ c", "C35R": "c₁ ↔ s₂",
       "B1R": "c ↔ carcasa", "B26": "s₂ ↔ carcasa"}
MARCHAS = {
    "N": [],
    "1": ["C1234", "B1R"],
    "2": ["C1234", "B26"],
    "3": ["C1234", "C35R"],
    "4": ["C1234", "C456"],
    "5": ["C456", "C35R"],
    "6": ["C456", "B26"],
    "R": ["C35R", "B1R"],
}
ETIQ_MARCHA = {"N": "N", "1": "1.ª", "2": "2.ª", "3": "3.ª", "4": "4.ª", "5": "5.ª", "6": "6.ª",
               "R": "R"}

COLORES = {
    "motor": "#444444", "ent": "#eb781e", "s1": "#9a9a9a", "c1": "#148228",
    "s2": "#c828b4", "s3": "#466ec8", "c": "#6e6e6e", "sal": "#1f9e89",
    "p_largo": "#f0c828", "p_corto": "#f6df84", "p1": "#f0c828",
    "on": "#d21e1e", "off": "#b0b0b0",
}


# ---------------------------------------------------------------------------
#  Cinemática y pares
# ---------------------------------------------------------------------------
def velocidades(z, marcha):
    """Velocidades normalizadas (ω_ent = 1). Incógnitas: ω_c1, ω_s2, ω_s3, ω_c, ω_sal."""
    rho1, rho2, rho3 = -z["s1"] / z["r1"], -z["s2"] / z["r2"], z["s3"] / z["r3"]
    idx = dict(c1=0, s2=1, s3=2, c=3, sal=4)
    A, b = [], []

    def ec(coef, rhs=0.0):
        fila = [0.0] * 5
        for k, v in coef.items():
            fila[idx[k]] += v
        A.append(fila)
        b.append(rhs)

    ec({"c1": rho1 - 1}, -1.0)                                  # (1) con ω_s1 = 0, ω_r1 = 1
    ec({"sal": 1, "c": -1 + rho2, "s2": -rho2})                 # (2)
    ec({"sal": 1, "c": -1 + rho3, "s3": -rho3})                 # (3)
    cond = {
        "C1234": ({"s3": 1, "c1": -1}, 0.0),
        "C35R": ({"s2": 1, "c1": -1}, 0.0),
        "C456": ({"c": 1}, 1.0),
        "B1R": ({"c": 1}, 0.0),
        "B26": ({"s2": 1}, 0.0),
    }
    if marcha == "N":            # punto muerto con el vehículo parado
        ec({"sal": 1}); ec({"c": 1})
    else:
        for e in MARCHAS[marcha]:
            ec(*cond[e])
    x = np.linalg.solve(np.array(A), np.array(b))
    w = {k: float(x[i]) for k, i in idx.items()}
    w.update(ent=1.0, s1=0.0)
    # satélites: absolutas y relativas al portasatélites
    w["p1_rel"] = (z["s1"] / z["p1"]) * w["c1"]                  # = -(z_s1/z_p1)(ω_s1 - ω_c1)
    w["p3d_rel"] = -(z["s3"] / z["p3d"]) * (w["s3"] - w["c"])
    w["p2_rel"] = (z["r2"] / z["p2"]) * (w["sal"] - w["c"])
    w["p1"] = w["c1"] + w["p1_rel"]
    w["p3d"] = w["c"] + w["p3d_rel"]
    w["p2"] = w["c"] + w["p2_rel"]
    return w


def pares(z, marcha):
    """Pares sin pérdidas con par de entrada unitario. None en punto muerto."""
    if marcha == "N":
        return None
    rho = [-z["s1"] / z["r1"], -z["s2"] / z["r2"], z["s3"] / z["r3"]]
    nombres = ["Ts1", "Tr1", "Tc1", "Ts2", "Tr2", "Tc2", "Ts3", "Tr3", "Tc3", "Tg", "TL"] + ELEMENTOS
    idx = {n: i for i, n in enumerate(nombres)}
    A, b = [], []

    def ec(coef, rhs=0.0):
        fila = [0.0] * len(nombres)
        for k, v in coef.items():
            fila[idx[k]] += v
        A.append(fila)
        b.append(rhs)

    for f, r in zip("123", rho):                 # equilibrio de cada fila y potencia relativa
        ec({f"Ts{f}": 1, f"Tr{f}": 1, f"Tc{f}": 1})
        ec({f"Ts{f}": 1, f"Tr{f}": r})
    ec({"Tr1": 1, "C456": 1}, 1.0)               # eje de entrada (T_ent = 1)
    ec({"Ts1": -1, "Tg": 1})                     # planeta s1 fijo
    ec({"Tc1": -1, "C1234": -1, "C35R": -1})     # portasatélites c1
    ec({"Ts3": -1, "C1234": 1})                  # planeta pequeño s3
    ec({"Ts2": -1, "C35R": 1, "B26": 1})         # planeta grande s2
    ec({"Tc2": -1, "Tc3": -1, "C456": 1, "B1R": 1})  # portasatélites c
    ec({"Tr2": -1, "Tr3": -1, "TL": 1})          # corona de salida
    for e in ELEMENTOS:
        if e not in MARCHAS[marcha]:
            ec({e: 1})
    x = np.linalg.solve(np.array(A), np.array(b))
    return {n: float(x[i]) for n, i in idx.items()}


def formulas(z):
    s1, r1, s2, r2, s3, r3 = z["s1"], z["r1"], z["s2"], z["r2"], z["s3"], z["r3"]
    a = r1 + s1
    return {
        "1": (r"\frac{(z_{r1}+z_{s1})\,z_{r3}}{z_{r1}\,z_{s3}}",
              rf"\frac{{{a}\cdot {r3}}}{{{r1}\cdot {s3}}}"),
        "2": (r"\frac{(z_{r1}+z_{s1})\,(z_{r2}z_{s3}+z_{r3}z_{s2})}{z_{r1}\,z_{s3}\,(z_{r2}+z_{s2})}",
              rf"\frac{{{a}\,({r2}\cdot {s3}+{r3}\cdot {s2})}}{{{r1}\cdot {s3}\cdot {r2 + s2}}}"),
        "3": (r"\frac{z_{r1}+z_{s1}}{z_{r1}}", rf"\frac{{{a}}}{{{r1}}}"),
        "4": (r"\frac{z_{r3}\,(z_{r1}+z_{s1})}{z_{r3}\,(z_{r1}+z_{s1})-z_{s1}\,z_{s3}}",
              rf"\frac{{{r3 * a}}}{{{r3 * a}-{s1 * s3}}}"),
        "5": (r"\frac{z_{r2}\,(z_{r1}+z_{s1})}{z_{r2}\,(z_{r1}+z_{s1})+z_{s1}\,z_{s2}}",
              rf"\frac{{{r2 * a}}}{{{r2 * a}+{s1 * s2}}}"),
        "6": (r"\frac{z_{r2}}{z_{r2}+z_{s2}}", rf"\frac{{{r2}}}{{{r2 + s2}}}"),
        "R": (r"-\frac{(z_{r1}+z_{s1})\,z_{r2}}{z_{r1}\,z_{s2}}",
              rf"-\frac{{{a}\cdot {r2}}}{{{r1}\cdot {s2}}}"),
    }


def coma(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def rpm(x):
    x = 0.0 if abs(x) < 0.5 else x
    return f"{x:,.0f}".replace(",", " ")


# ---------------------------------------------------------------------------
#  Barra lateral
# ---------------------------------------------------------------------------
st.sidebar.header("Funcionamiento")
n_in = st.sidebar.number_input("Velocidad de la turbina (rpm)", 0.0, 8000.0, 2000.0, 100.0,
                               format="%.0f")
m_in = st.sidebar.number_input("Par de entrada (N·m)", 0.0, 1000.0, 300.0, 10.0, format="%.0f")
lockup = st.sidebar.checkbox("Embrague anulador (lock-up) cerrado", value=True,
                             help="Solo cambia el dibujo: la cinemática parte de la "
                                  "velocidad de la turbina, que es la entrada de la caja.")
st.sidebar.header("Animación")
vel_anim = st.sidebar.slider("Velocidad de la animación", 0.1, 3.0, 1.0, 0.1)
ver_s2 = st.sidebar.checkbox("Dibujar el planeta grande s₂ (está en otro plano)", value=True)

with st.sidebar.expander("Números de dientes"):
    z = {}
    for k, txt in [("s1", "z_s1 planeta fila 1"), ("r1", "z_r1 corona fila 1"),
                   ("p1", "z_p1 satélite fila 1"), ("s2", "z_s2 planeta grande"),
                   ("r2", "z_r2 = z_r3 corona"), ("p2", "z_p2 = z_p3b satélite largo"),
                   ("s3", "z_s3 planeta pequeño"), ("p3d", "z_p3d satélite corto")]:
        z[k] = int(st.number_input(txt, 8, 150, Z0[k], 1, key="z_" + k))
    z["r3"] = z["r2"]
    st.caption("Valores del artículo: 37/71/17 · 38/85/23 · 31/85/23/28.")

# ---------------------------------------------------------------------------
#  Cabecera y selector de marcha
# ---------------------------------------------------------------------------
st.title("Caja automática ZF 6HP26")
st.caption("Disposición Lepelletier: tren simple con el planeta fijo + tren Ravigneaux. "
           "Rayas blancas en movimiento: giro de cada eje.")

marcha = st.radio("Marcha", list(MARCHAS), index=1, horizontal=True,
                  format_func=lambda m: ETIQ_MARCHA[m])

w = velocidades(z, marcha)
T = pares(z, marcha)
i_rel = 1.0 / w["sal"] if abs(w["sal"]) > 1e-12 else math.inf

# ---------------------------------------------------------------------------
#  Animación
# ---------------------------------------------------------------------------
param = {
    "z": z, "w": w, "activos": MARCHAS[marcha], "marcha": ETIQ_MARCHA[marcha],
    "i": coma(i_rel) if math.isfinite(i_rel) else "—",
    "anim": vel_anim, "ver_s2": ver_s2, "lockup": lockup, "col": COLORES,
}

HTML = r"""
<div style="font-family:sans-serif">
  <canvas id="c" width="1200" height="560" style="width:100%;max-width:1200px"></canvas>
  <div style="margin-top:2px">
    <button id="bp">Pausa</button> <button id="br">Reiniciar</button>
  </div>
</div>
<script>
const P = __PARAM__;
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const W = cv.width, H = cv.height, Y0 = 290, TAU = 2 * Math.PI;
const C = P.col, z = P.z, w = P.w;
const frac = x => x - Math.floor(x);
const fmt = x => (Math.abs(x) < 5e-4 ? 0 : x).toFixed(3).replace('.', ',');
const on = e => P.activos.includes(e);

// ---------------- geometría ----------------
const A = {x: 330, y: Y0}, B = {x: 900, y: Y0};
const m1 = 116 / (z.r1 / 2 + 4), m2 = 166 / (z.r2 / 2 + 4);
const R1 = m1 * (z.r1 / 2 + 4), R2 = m2 * (z.r2 / 2 + 4);
const rs1 = m1 * z.s1 / 2, rr1 = m1 * z.r1 / 2, rp1 = m1 * z.p1 / 2, d1 = (rs1 + rr1) / 2;
const rs2 = m2 * z.s2 / 2, rs3 = m2 * z.s3 / 2, rr = m2 * z.r2 / 2;
const rpl = m2 * z.p2 / 2, rps = m2 * z.p3d / 2;
const dl = rr - rpl, ds = rs3 + rps, dls = rpl + rps;
const th = Math.acos(Math.max(-1, Math.min(1, (ds * ds + dl * dl - dls * dls) / (2 * ds * dl))));

// ---------------- fases para que engranen ----------------
const ext = (beta, phA, zA, zB) =>               // B engrana por fuera con A en la dirección beta
  beta + Math.PI - (0.5 - frac((beta - phA) * zA / TAU)) * TAU / zB;
const N = 3;
const beta1 = [...Array(N).keys()].map(k => Math.PI / 2 + k * TAU / N);
const ph = {s1: 0};
ph.p1 = beta1.map(b => ext(b, 0, z.s1, z.p1));
ph.r1 = beta1[0] - (frac((beta1[0] - ph.p1[0]) * z.p1 / TAU) + 0.5) * TAU / z.r1;
const sig = [...Array(N).keys()].map(k => Math.PI / 2 + 0.3 + k * TAU / N);   // satélites cortos
const lam = sig.map(s => s + th);                                              // satélites largos
ph.s3 = 0;
ph.p3d = sig.map(s => ext(s, 0, z.s3, z.p3d));
ph.p2 = sig.map((s, k) => {
  const g = Math.atan2(dl * Math.sin(lam[k]) - ds * Math.sin(s), dl * Math.cos(lam[k]) - ds * Math.cos(s));
  return ext(g, ph.p3d[k], z.p3d, z.p2);
});
ph.r = lam[0] - (frac((lam[0] - ph.p2[0]) * z.p2 / TAU) + 0.5) * TAU / z.r2;
ph.s2 = lam[0] - (0.5 - frac((lam[0] + Math.PI - ph.p2[0]) * z.p2 / TAU)) * TAU / z.s2;

// ---------------- utilidades de dibujo ----------------
function gear(zz, r, rot, internal, m, nuevo = true) {
  const p = TAU / zz, ha = m, hf = 1.25 * m;
  const tip = internal ? r - ha : r + ha, root = internal ? r + hf : r - hf;
  if (nuevo) ctx.beginPath();
  for (let j = 0; j < zz; j++) {
    const a = rot + j * p;
    [[root, a - 0.5 * p], [root, a - 0.27 * p], [tip, a - 0.13 * p],
     [tip, a + 0.13 * p], [root, a + 0.27 * p]].forEach(([rr_, aa], k) => {
      const x = rr_ * Math.cos(aa), y = rr_ * Math.sin(aa);
      (j === 0 && k === 0) ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
  }
  ctx.closePath();
}
function hatchPattern() {
  const c = document.createElement('canvas'); c.width = c.height = 8;
  const g = c.getContext('2d'); g.strokeStyle = 'rgba(0,0,0,0.45)'; g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, 8); g.lineTo(8, 0); g.stroke();
  return ctx.createPattern(c, 'repeat');
}
const pat = hatchPattern();
function stripe(r0, r1_, a, lw = 3) {
  ctx.strokeStyle = 'white'; ctx.lineWidth = lw; ctx.beginPath();
  ctx.moveTo(r0 * Math.cos(a), r0 * Math.sin(a)); ctx.lineTo(r1_ * Math.cos(a), r1_ * Math.sin(a));
  ctx.stroke();
}
function text(t, x, y, opt = {}) {
  ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.font = (opt.bold ? 'bold ' : '') + (opt.size || 12) + 'px sans-serif';
  ctx.textAlign = opt.align || 'center'; ctx.textBaseline = 'middle';
  ctx.lineWidth = 4; ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.strokeText(t, x, y);
  ctx.fillStyle = opt.color || '#222'; ctx.fillText(t, x, y);
  ctx.restore();
}
// eje (polilínea) con rayas que se mueven según su velocidad
function shaft(pts, color, wv, fijo = false) {
  ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.lineJoin = 'round'; ctx.lineCap = 'butt';
  ctx.beginPath(); pts.forEach(([x, y], k) => k ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
  ctx.strokeStyle = color; ctx.lineWidth = 7; ctx.stroke();
  if (!fijo) {
    ctx.setLineDash([9, 11]); ctx.lineDashOffset = -wv * t * 60;
    ctx.strokeStyle = 'rgba(255,255,255,0.95)'; ctx.lineWidth = 2.5; ctx.stroke();
  }
  ctx.restore();
}
function ground(x, y, dir) {            // dir = -1: masa por encima, +1: por debajo
  ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = pat; ctx.fillRect(x - 24, dir < 0 ? y - 12 : y, 48, 12);
  ctx.strokeStyle = '#222'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x - 24, y); ctx.lineTo(x + 24, y); ctx.stroke();
  ctx.restore();
}
// embrague de discos sobre un eje horizontal
function clutch(x, y, cL, cR, act, lab, labDy = -30) {
  ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0);
  const g = act ? 5 : 15, col = act ? C.on : C.off;
  ctx.lineWidth = 7;
  ctx.strokeStyle = cL; ctx.beginPath(); ctx.moveTo(x - 14, y); ctx.lineTo(x - g / 2, y); ctx.stroke();
  ctx.strokeStyle = cR; ctx.beginPath(); ctx.moveTo(x + g / 2, y); ctx.lineTo(x + 14, y); ctx.stroke();
  ctx.fillStyle = col;
  ctx.fillRect(x - g / 2 - 4, y - 19, 4, 38); ctx.fillRect(x + g / 2, y - 19, 4, 38);
  ctx.restore();
  text(lab, x, y + labDy, {bold: act, color: act ? C.on : '#555', size: 13});
}
// freno: del eje (x, y) hacia la carcasa en la dirección dir
function brake(x, y, dir, cEl, act, lab) {
  ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0);
  const g = act ? 5 : 13, col = act ? C.on : C.off, y1 = y + dir * 14;
  ctx.strokeStyle = cEl; ctx.lineWidth = 5;
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y1); ctx.stroke();
  ctx.fillStyle = col;
  ctx.fillRect(x - 20, dir > 0 ? y1 : y1 - 4, 40, 4);
  const y2 = y1 + dir * (g + 4);
  ctx.fillRect(x - 20, dir > 0 ? y2 - 4 : y2, 40, 4);
  ctx.strokeStyle = '#222'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x, y2); ctx.lineTo(x, y2 + dir * 8); ctx.stroke();
  ctx.restore();
  ground(x, y2 + dir * 8, dir);
  text(lab, x + 38, y2 - dir * 2, {bold: act, color: act ? C.on : '#555', size: 13});
}

let t = 0, run = true, last = null;
document.getElementById('bp').onclick = e => { run = !run; e.target.textContent = run ? 'Pausa' : 'Seguir'; };
document.getElementById('br').onclick = () => { t = 0; };
const K = TAU * 0.3;                     // la turbina gira a 0,3 vueltas/s (×velocidad de animación)
const ang = (k, fase) => fase + K * w[k] * t;

// ---------------- convertidor de par ----------------
function convertidor() {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  // estator a la carcasa (por debajo)
  ctx.strokeStyle = '#2e7d32'; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(105, Y0 + 12); ctx.lineTo(105, Y0 + 125); ctx.stroke();
  ground(105, Y0 + 125, 1);
  // bomba (motor) y turbina
  [-1, 1].forEach(s => {
    ctx.beginPath(); ctx.ellipse(100, Y0 + s * 50, 26, 38, 0, Math.PI / 2, 3 * Math.PI / 2);
    ctx.closePath(); ctx.fillStyle = '#f2cf3a'; ctx.fill(); ctx.strokeStyle = '#7a6400'; ctx.lineWidth = 1.2; ctx.stroke();
    ctx.beginPath(); ctx.ellipse(110, Y0 + s * 50, 26, 38, 0, -Math.PI / 2, Math.PI / 2);
    ctx.closePath(); ctx.fillStyle = '#f8e48c'; ctx.fill(); ctx.stroke();
  });
  ctx.fillStyle = '#4caf50'; ctx.fillRect(100, Y0 - 14, 10, 28);
  ctx.strokeStyle = '#1b5e20'; ctx.lineWidth = 1; ctx.strokeRect(100, Y0 - 14, 10, 28);
  text('Convertidor', 105, Y0 + 152, {size: 12, color: '#555'});
}

// ---------------- tren simple (fila 1) ----------------
function fila1() {
  ctx.setTransform(1, 0, 0, -1, A.x, A.y);
  const thc = K * w.c1 * t;
  // corona r1 (entrada)
  ctx.beginPath(); ctx.moveTo(R1, 0); ctx.arc(0, 0, R1, 0, TAU); ctx.closePath();
  gear(z.r1, rr1, ang('ent', ph.r1), true, m1, false);
  ctx.fillStyle = C.ent; ctx.fill('evenodd'); ctx.strokeStyle = '#333'; ctx.lineWidth = 0.8; ctx.stroke();
  stripe(rr1 + 1.3 * m1, R1 - 0.3 * m1, ang('ent', 0), 4);
  // portasatélites c1 (placa translúcida)
  ctx.globalAlpha = 0.35; ctx.fillStyle = C.c1; ctx.beginPath();
  beta1.forEach((b, k) => { const x = d1 * Math.cos(b + thc), y = d1 * Math.sin(b + thc); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
  // satélites
  beta1.forEach((b, k) => {
    ctx.save(); ctx.translate(d1 * Math.cos(b + thc), d1 * Math.sin(b + thc));
    gear(z.p1, rp1, ang('p1', ph.p1[k]), false, m1);
    ctx.fillStyle = C.p1; ctx.fill(); ctx.strokeStyle = '#333'; ctx.lineWidth = 0.8; ctx.stroke();
    stripe(0.25 * rp1, 0.8 * rp1, ang('p1', ph.p1[k]), 2.5);
    ctx.fillStyle = C.c1; ctx.beginPath(); ctx.arc(0, 0, 0.22 * rp1, 0, TAU); ctx.fill();
    ctx.restore();
  });
  // planeta s1 fijo
  gear(z.s1, rs1, ph.s1, false, m1);
  ctx.fillStyle = C.s1; ctx.fill(); ctx.fillStyle = pat; ctx.fill();
  ctx.strokeStyle = '#333'; ctx.lineWidth = 0.8; ctx.stroke();
  ctx.fillStyle = C.c1; ctx.beginPath(); ctx.arc(0, 0, 0.25 * rs1, 0, TAU); ctx.fill();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  text('s₁ (' + z.s1 + ')', A.x, A.y + 0.55 * rs1, {size: 11});
  text('Fila 1: tren simple', A.x, A.y + R1 + 20, {size: 13, bold: true, color: '#333'});
}

// ---------------- Ravigneaux (filas 2 y 3) ----------------
function ravigneaux() {
  ctx.setTransform(1, 0, 0, -1, B.x, B.y);
  const thc = K * w.c * t;
  const fijoC = on('B1R'), fijoS2 = on('B26');
  // planeta grande s2 (otro plano axial)
  if (P.ver_s2) {
    ctx.globalAlpha = 0.45;
    gear(z.s2, rs2, ang('s2', ph.s2), false, m2);
    ctx.fillStyle = C.s2; ctx.fill(); if (fijoS2) { ctx.fillStyle = pat; ctx.fill(); }
    ctx.globalAlpha = 1; ctx.setLineDash([5, 4]); ctx.strokeStyle = C.s2; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.setLineDash([]);
  }
  // portasatélites común c
  ctx.globalAlpha = 0.3; ctx.fillStyle = C.c; ctx.beginPath();
  sig.forEach((s, k) => {
    const a = [ds * Math.cos(s + thc), ds * Math.sin(s + thc)], b = [dl * Math.cos(lam[k] + thc), dl * Math.sin(lam[k] + thc)];
    k ? ctx.lineTo(...a) : ctx.moveTo(...a); ctx.lineTo(...b);
  });
  ctx.closePath(); ctx.fill(); if (fijoC) { ctx.fillStyle = pat; ctx.fill(); } ctx.globalAlpha = 1;
  // satélites cortos (en el plano de s3, detrás de la corona)
  sig.forEach((s, k) => {
    ctx.save(); ctx.translate(ds * Math.cos(s + thc), ds * Math.sin(s + thc));
    gear(z.p3d, rps, ang('p3d', ph.p3d[k]), false, m2);
    ctx.fillStyle = C.p_corto; ctx.fill(); ctx.strokeStyle = '#333'; ctx.lineWidth = 0.8; ctx.stroke();
    stripe(0.25 * rps, 0.8 * rps, ang('p3d', ph.p3d[k]), 2.5);
    ctx.fillStyle = C.c; ctx.beginPath(); ctx.arc(0, 0, 0.2 * rps, 0, TAU); ctx.fill();
    ctx.restore();
  });
  // corona de salida
  ctx.beginPath(); ctx.moveTo(R2, 0); ctx.arc(0, 0, R2, 0, TAU); ctx.closePath();
  gear(z.r2, rr, ang('sal', ph.r), true, m2, false);
  ctx.fillStyle = C.sal; ctx.fill('evenodd'); ctx.strokeStyle = '#333'; ctx.lineWidth = 0.8; ctx.stroke();
  stripe(rr + 1.3 * m2, R2 - 0.3 * m2, ang('sal', 0), 4);
  // satélites largos (engranan con la corona, con s2 y con los cortos)
  sig.forEach((s, k) => {
    ctx.save(); ctx.translate(dl * Math.cos(lam[k] + thc), dl * Math.sin(lam[k] + thc));
    gear(z.p2, rpl, ang('p2', ph.p2[k]), false, m2);
    ctx.fillStyle = C.p_largo; ctx.fill(); ctx.strokeStyle = '#333'; ctx.lineWidth = 0.8; ctx.stroke();
    stripe(0.25 * rpl, 0.8 * rpl, ang('p2', ph.p2[k]), 2.5);
    ctx.fillStyle = C.c; ctx.beginPath(); ctx.arc(0, 0, 0.2 * rpl, 0, TAU); ctx.fill();
    ctx.restore();
  });
  // planeta pequeño s3
  gear(z.s3, rs3, ang('s3', ph.s3), false, m2);
  ctx.fillStyle = C.s3; ctx.fill(); ctx.strokeStyle = '#333'; ctx.lineWidth = 0.8; ctx.stroke();
  stripe(0.3 * rs3, 0.85 * rs3, ang('s3', ph.s3), 4);
  ctx.fillStyle = 'white'; ctx.beginPath(); ctx.arc(0, 0, 0.18 * rs3, 0, TAU); ctx.fill();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  text('s₃ (' + z.s3 + ')', B.x, B.y + 0.5 * rs3, {size: 11});
  text('Filas 2 y 3: tren Ravigneaux', B.x - 110, B.y + R2 + 20, {size: 13, bold: true, color: '#333'});
  // leyenda de satélites
  const lx = 962, ly = 30;
  [[C.p_largo, 'satélite largo p₂ = p₃b (' + z.p2 + ')'], [C.p_corto, 'satélite corto p₃d (' + z.p3d + ')'],
   [C.s2, 'planeta grande s₂ (' + z.s2 + '), detrás']].forEach(([c, s], k) => {
    ctx.fillStyle = c; ctx.fillRect(lx, ly + 18 * k - 6, 12, 12);
    text(s, lx + 18, ly + 18 * k, {size: 11, align: 'left', color: '#333'});
  });
}

// ---------------- ejes, embragues y frenos ----------------
const XJ = 540, XC = 640, YT = 70, YB = 505;
function ejes() {
  // motor, bomba, turbina
  shaft([[0, Y0], [70, Y0], [70, Y0 - 50]], C.motor, w.ent);
  shaft([[70, Y0], [70, Y0 + 50]], C.motor, w.ent);
  shaft([[140, Y0 - 50], [140, Y0 + 50]], C.ent, w.ent);
  shaft([[140, Y0], [A.x - R1, Y0]], C.ent, w.ent);
  // lock-up
  shaft([[45, Y0], [45, Y0 - 115], [95, Y0 - 115]], C.motor, w.ent);
  shaft([[115, Y0 - 115], [165, Y0 - 115], [165, Y0]], C.ent, w.ent);
  clutch(105, Y0 - 115, C.motor, C.ent, P.lockup, 'lock-up', -30);
  // entrada → C456 → portasatélites c
  shaft([[190, Y0], [190, YB], [XC - 14, YB]], C.ent, w.ent);
  shaft([[XC + 14, YB], [B.x, YB], [B.x, B.y + R2]], C.c, w.c, on('B1R'));
  clutch(XC, YB, C.ent, C.c, on('C456'), 'C₄,₅,₆', 32);
  brake(780, YB, 1, C.c, on('B1R'), 'B₁,R');
  // planeta s1 a la carcasa
  shaft([[A.x, A.y - R1], [A.x, 128]], C.s1, 0, true);
  ground(A.x, 128, -1);
  // portasatélites c1 → C1234 → s3   y   c1 → C35R → s2
  shaft([[A.x + R1, Y0], [XJ, Y0], [XC - 14, Y0]], C.c1, w.c1);
  shaft([[XJ, Y0], [XJ, YT], [XC - 14, YT]], C.c1, w.c1);
  shaft([[XC + 14, Y0], [B.x - R2, Y0]], C.s3, w.s3);
  shaft([[XC + 14, YT], [B.x, YT], [B.x, B.y - R2]], C.s2, w.s2, on('B26'));
  clutch(XC, Y0, C.c1, C.s3, on('C1234'), 'C₁,₂,₃,₄', 32);
  clutch(XC, YT, C.c1, C.s2, on('C35R'), 'C₃,₅,R', -30);
  brake(780, YT, -1, C.s2, on('B26'), 'B₂,₆');
  // salida
  shaft([[B.x + R2, Y0], [W, Y0]], C.sal, w.sal);
}

function rotulos() {
  const r = (s, x, y, c, al = 'center') => text(s, x, y, {size: 12, color: c, align: al, bold: true});
  r('motor', 22, Y0 - 16, C.motor);
  r('r₁ · ' + fmt(w.ent), A.x - 0.72 * R1, A.y - 0.86 * R1, C.ent, 'right');
  r('s₁ · 0', A.x + 10, 150, C.s1, 'left');
  r('c₁ · ' + fmt(w.c1), A.x + R1 + 8, Y0 + 18, C.c1, 'left');
  r('s₃ · ' + fmt(w.s3), B.x - R2 - 6, Y0 + 18, C.s3, 'right');
  r('s₂ · ' + fmt(w.s2), B.x + 10, B.y - R2 - 22, C.s2, 'left');
  r('c · ' + fmt(w.c), B.x + 10, B.y + R2 + 40, C.c, 'left');
  r('salida · ' + fmt(w.sal), W - 6, Y0 - 16, C.sal, 'right');
  text('ω / ω_ent', W - 6, Y0 + 16, {size: 11, align: 'right', color: '#777'});
  text('Marcha ' + P.marcha, 14, 26, {size: 22, bold: true, align: 'left', color: '#222'});
  text('i = ω_ent / ω_sal = ' + P.i, 14, 54, {size: 16, align: 'left', color: C.on, bold: true});
}

function draw() {
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, W, H);
  convertidor();
  fila1();
  ravigneaux();
  ejes();
  rotulos();
}
function loop(ts) {
  if (last !== null && run) t += (ts - last) / 1000 * P.anim;
  last = ts; draw(); requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
</script>
"""


def incrustar_html(html, alto):
    """st.iframe en versiones recientes de Streamlit; components.html en las antiguas."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=alto)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=alto)


incrustar_html(HTML.replace("__PARAM__", json.dumps(param)), 600)

# ---------------------------------------------------------------------------
#  Resultados
# ---------------------------------------------------------------------------
col_rel, col_vel, col_mando = st.columns([1.1, 1, 1.2])

with col_rel:
    st.subheader("Relación de cambio")
    if marcha == "N":
        st.info("Punto muerto: ningún elemento de la caja Ravigneaux está aplicado; con el "
                "vehículo parado solo gira el tren de entrada.")
    else:
        sym, num = formulas(z)[marcha]
        sub = "R" if marcha == "R" else marcha
        st.latex(rf"i_{{{sub}}} = {sym}")
        st.latex(rf"i_{{{sub}}} = {num} = \mathbf{{{coma(i_rel)}}}".replace(".", "{,}"))
        c1, c2 = st.columns(2)
        c1.metric("Velocidad de salida", f"{rpm(n_in * w['sal'])} rpm")
        c2.metric("Par de salida", f"{rpm(m_in * i_rel)} N·m")
        if marcha == "3":
            st.caption("Con C₁,₂,₃,₄ y C₃,₅,R aplicados s₂ = s₃: el Ravigneaux gira como un "
                       "sólido rígido y solo queda la reducción de la fila 1.")
        if marcha == "6":
            st.caption("La fila 1 gira en vacío: la relación no depende de z_s1 ni de z_r1.")

with col_vel:
    st.subheader("Velocidades (rpm)")
    filas = [
        ("Entrada, corona r₁", w["ent"], None),
        ("Planeta s₁ (fijo)", w["s1"], None),
        ("Portasatélites c₁", w["c1"], None),
        ("Satélite p₁", w["p1"], w["p1_rel"]),
        ("Planeta grande s₂", w["s2"], None),
        ("Planeta pequeño s₃", w["s3"], None),
        ("Portasatélites c", w["c"], None),
        ("Satélite largo p₂ = p₃b", w["p2"], w["p2_rel"]),
        ("Satélite corto p₃d", w["p3d"], w["p3d_rel"]),
        ("Salida, corona r₂ = r₃", w["sal"], None),
    ]
    datos = [{"Elemento": n, "Absoluta": rpm(n_in * v),
              "Relativa al portasat.": "" if r is None else rpm(n_in * r)} for n, v, r in filas]
    st.dataframe(datos, hide_index=True, width="stretch")

with col_mando:
    st.subheader("Elementos de mando")
    lado = {"C1234": ("c1", "s3"), "C456": ("ent", "c"), "C35R": ("c1", "s2"),
            "B1R": (None, "c"), "B26": (None, "s2")}
    datos = []
    for e in ELEMENTOS:
        a, b = lado[e]
        dw = w[b] - (w[a] if a else 0.0)
        act = e in MARCHAS[marcha]
        par = T[e] * m_in if (T and act) else None
        datos.append({"Elemento": NOMBRE_EL[e], "Une": UNE[e],
                      "Estado": "● aplicado" if act else "libre",
                      "Δω (rpm)": "0" if act else rpm(n_in * dw),
                      "Par (N·m)": "" if par is None else rpm(par)})
    st.dataframe(datos, hide_index=True, width="stretch")
    if T:
        carcasa = T["Tg"] + T["B1R"] + T["B26"]
        st.caption(f"Δω: deslizamiento entre los discos de los elementos libres. "
                   f"Par con signo: positivo en el sentido del par de entrada. "
                   f"Par en la carcasa (s₁ + frenos): {rpm(carcasa * m_in)} N·m = "
                   f"(i − 1)·M_ent. Par en s₁: {rpm(T['Tg'] * m_in)} N·m.")

# ---------------------------------------------------------------------------
#  Ecuaciones y tabla de marchas
# ---------------------------------------------------------------------------
with st.expander("Ecuaciones de Willis y condiciones de cada elemento"):
    c1, c2 = st.columns(2)
    with c1:
        st.latex(r"(1)\quad \frac{\omega_{r1}-\omega_{c1}}{\omega_{s1}-\omega_{c1}} = -\frac{z_{s1}}{z_{r1}}")
        st.latex(r"(2)\quad \frac{\omega_{sal}-\omega_{c}}{\omega_{s2}-\omega_{c}} = -\frac{z_{s2}}{z_{r2}}")
        st.latex(r"(3)\quad \frac{\omega_{sal}-\omega_{c}}{\omega_{s3}-\omega_{c}} = "
                 r"\Bigl(-\frac{z_{s3}}{z_{p3d}}\Bigr)\Bigl(-\frac{z_{p3d}}{z_{p3b}}\Bigr)"
                 r"\Bigl(+\frac{z_{p3b}}{z_{r3}}\Bigr) = +\frac{z_{s3}}{z_{r3}}")
    with c2:
        st.markdown(
            "- **C₁,₂,₃,₄**: $\\omega_{s3}=\\omega_{c1}$  \n"
            "- **C₃,₅,R**: $\\omega_{s2}=\\omega_{c1}$  \n"
            "- **C₄,₅,₆**: $\\omega_{c}=\\omega_{ent}$  \n"
            "- **B₁,R**: $\\omega_{c}=0$  \n"
            "- **B₂,₆**: $\\omega_{s2}=0$  \n\n"
            "Cada marcha aplica dos elementos: con ω_ent y $\\omega_{s1}=0$ quedan cinco "
            "ecuaciones para ω_c1, ω_s2, ω_s3, ω_c y ω_sal."
        )

st.subheader("Todas las marchas")
tabla = []
prev = None
for m in ["1", "2", "3", "4", "5", "6", "R"]:
    ii = 1.0 / velocidades(z, m)["sal"]
    fila = {"Marcha": ETIQ_MARCHA[m]}
    for e in ELEMENTOS:
        fila[NOMBRE_EL[e]] = "●" if e in MARCHAS[m] else ""
    fila["i"] = coma(ii)
    fila["Salto i_k/i_k+1"] = coma(prev / ii, 2) if (prev and m != "R") else ""
    fila[""] = "◀" if m == marcha else ""
    tabla.append(fila)
    prev = ii
st.dataframe(tabla, hide_index=True, width="stretch")
i1, i6 = 1 / velocidades(z, "1")["sal"], 1 / velocidades(z, "6")["sal"]
st.caption(f"Apertura de la caja i₁/i₆ = {coma(i1 / i6, 2)}. "
           f"Montaje fila 1: z_s1 + 2 z_p1 = {z['s1'] + 2 * z['p1']} (z_r1 = {z['r1']}); "
           f"fila 2: z_s2 + 2 z_p2 = {z['s2'] + 2 * z['p2']} (z_r2 = {z['r2']}).")
