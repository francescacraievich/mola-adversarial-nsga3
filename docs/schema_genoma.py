from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT = Path(__file__).resolve().parent

# geni: (indice, etichetta, gruppo parametro)
genes = [
    (0, "g0", "noise_direction"), (1, "g1", "noise_direction"), (2, "g2", "noise_direction"),
    (3, "g3", "noise_intensity"),
    (4, "g4", "curvature_strength"),
    (5, "g5", "dropout_rate"),
    (6, "g6", "ghost_ratio"),
    (7, "g7", "cluster_direction"), (8, "g8", "cluster_direction"), (9, "g9", "cluster_direction"),
    (10, "g10", "cluster_strength"),
    (11, "g11", "spatial_correlation"),
    (12, "g12", "geometric_distortion"),
    (13, "g13", "edge_attack_strength"),
    (14, "g14", "temporal_drift_strength"),
    (15, "g15", "scanline_strength"),
    (16, "g16", "strategic_ghost"),
]
# parametri: (nome, etichetta con range)
params = [
    ("noise_direction",        "noise_direction  (versore 3D)"),
    ("noise_intensity",        "noise_intensity  [0, 5 cm]"),
    ("curvature_strength",     "curvature_strength  [0, 1]"),
    ("dropout_rate",           "dropout_rate  [0, 15 %]"),
    ("ghost_ratio",            "ghost_ratio  [0, 2 %]"),
    ("cluster_direction",      "cluster_direction  (versore 3D)"),
    ("cluster_strength",       "cluster_strength  [0, 1]"),
    ("spatial_correlation",    "spatial_correlation  [0, 1]"),
    ("geometric_distortion",   "geometric_distortion  [0, 1]"),
    ("edge_attack_strength",   "edge_attack_strength  [0, 1]"),
    ("temporal_drift_strength","temporal_drift_strength  [0, 1]"),
    ("scanline_strength",      "scanline_strength  [0, 1]"),
    ("strategic_ghost",        "strategic_ghost  [0, 1]"),
]
# operatori in ordine di applicazione: (num, nome, sottotitolo, parametri principali, parametri riusati, globale?)
ops = [
    (1, "Maschera di curvatura", "pesi 1.0 sui punti ad alta curvatura, 0.3 altrove",
        ["curvature_strength"], [], False),
    (2, "Rumore per punto", "gaussiano, ≤ 5 cm, con bias direzionale × maschera",
        ["noise_direction", "noise_intensity", "spatial_correlation"], [], False),
    (3, "Spostamento di cluster", "5 zone casuali, ≤ 3 cm, decadimento esponenziale",
        ["cluster_direction", "cluster_strength"], [], False),
    (4, "Dropout", "rimozione ≤ 15 %, più probabile nelle zone dense",
        ["dropout_rate"], [], False),
    (5, "Ghost points", "≤ 2 % di punti falsi, casuali o ancorati alle feature",
        ["ghost_ratio", "strategic_ghost"], [], False),
    (6, "Distorsione geometrica", "bias ≤ 5 cm, yaw ≤ 2.9°, scala ≤ 3 %  — GLOBALE",
        ["geometric_distortion"], ["noise_direction", "cluster_direction"], True),
    (7, "Attacco ai bordi", "500 punti spigolo, ≤ 8 cm perpendicolari alla direzione principale",
        ["edge_attack_strength"], [], False),
    (8, "Drift temporale", "trasla l'intera nuvola, accumula ≤ 5 cm/frame  — GLOBALE",
        ["temporal_drift_strength"], ["noise_direction"], True),
    (9, "Scanline", "spostamento lungo il raggio laser, 3 cm + onda 2 cm",
        ["scanline_strength"], [], False),
]

fig, ax = plt.subplots(figsize=(15, 11))
ax.set_xlim(0, 15.6); ax.set_ylim(-0.5, 11.7); ax.axis("off")

C_GENE, C_PAR, C_OP, C_GLOB = "#dfe7f2", "#e9f1e4", "#fff1d6", "#f7d9d0"
EDGE = "#4a5568"

# ---- colonna geni
x0, w0, h0 = 0.4, 1.0, 0.44
top, bottom = 10.2, 0.8
gene_y = {}
step = (top - bottom) / (len(genes) - 1)
for i, (idx, lab, grp) in enumerate(genes):
    y = top - i * step
    gene_y[idx] = y
    ax.add_patch(FancyBboxPatch((x0, y - h0 / 2), w0, h0, boxstyle="round,pad=0.02",
                                fc=C_GENE, ec=EDGE, lw=0.8))
    ax.text(x0 + w0 / 2, y, lab, ha="center", va="center", fontsize=9.5, family="monospace")

# ---- colonna parametri
x1, w1, h1 = 3.0, 4.2, 0.5
par_y = {}
pstep = (top - bottom) / (len(params) - 1)
for i, (name, lab) in enumerate(params):
    y = top - i * pstep
    par_y[name] = y
    ax.add_patch(FancyBboxPatch((x1, y - h1 / 2), w1, h1, boxstyle="round,pad=0.02",
                                fc=C_PAR, ec=EDGE, lw=0.8))
    ax.text(x1 + 0.15, y, lab, ha="left", va="center", fontsize=9.5, family="monospace")

# geni -> parametri
for idx, lab, grp in genes:
    ax.plot([x0 + w0, x1], [gene_y[idx], par_y[grp]], color="#8a94a6", lw=0.9)

# ---- colonna operatori
x2, w2, h2 = 9.3, 5.4, 0.86
op_y = {}
ostep = (top - bottom) / (len(ops) - 1)
for i, (n, name, sub, main, reused, glob) in enumerate(ops):
    y = top - i * ostep
    op_y[n] = y
    ax.add_patch(FancyBboxPatch((x2, y - h2 / 2), w2, h2, boxstyle="round,pad=0.02",
                                fc=C_GLOB if glob else C_OP, ec=EDGE, lw=1.0))
    ax.text(x2 + 0.15, y + 0.18, f"{n}. {name}", ha="left", va="center", fontsize=11,
            fontweight="bold")
    ax.text(x2 + 0.15, y - 0.2, sub, ha="left", va="center", fontsize=8.6, color="#333")
    for p in main:
        ax.plot([x1 + w1, x2], [par_y[p], y], color="#6b7280", lw=1.0)

# maschera -> rumore
ax.add_patch(FancyArrowPatch((x2 + 0.9, op_y[1] - h2 / 2), (x2 + 0.9, op_y[2] + h2 / 2),
                             arrowstyle="-|>", mutation_scale=12, color=EDGE, lw=1.2))
ax.text(x2 + 1.0, (op_y[1] + op_y[2]) / 2, "pesa il rumore", fontsize=8, va="center", color=EDGE)

# ordine di applicazione: freccia verticale a sinistra degli operatori
ax.add_patch(FancyArrowPatch((x2 + w2 + 0.3, top + 0.45), (x2 + w2 + 0.3, bottom - 0.45),
                             arrowstyle="-|>", mutation_scale=14, color=EDGE, lw=1.2))
ax.text(x2 + w2 + 0.45, (top + bottom) / 2, "ordine di applicazione a ogni scan", rotation=270,
        ha="center", va="center", fontsize=8.5, color=EDGE)

# titoli
ax.text(x0 + w0 / 2, 11.25, "17 geni\n∈ [−1, 1]", ha="center", va="center", fontsize=11, fontweight="bold")
ax.text(x1 + w1 / 2, 11.25, "13 parametri\n11 scalari + 2 direzioni (3 geni ciascuna)", ha="center",
        va="center", fontsize=11, fontweight="bold")
ax.text(x2 + w2 / 2, 11.25, "9 operatori\napplicati in sequenza alla nuvola", ha="center",
        va="center", fontsize=11, fontweight="bold")

# legenda
ax.plot([0.5, 1.1], [-0.1, -0.1], color="#6b7280", lw=1.0)
ax.text(1.2, -0.1, "parametro proprio dell'operatore", va="center", fontsize=8.5)

fig.savefig(OUT / "schema_genoma.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "schema_genoma.svg", bbox_inches="tight", facecolor="white")
print("ok")
