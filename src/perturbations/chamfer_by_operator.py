#!/usr/bin/env python3
"""
Contributo di ogni operatore alla percettibilita' (distanza di Chamfer).

Prende una nuvola reale (dal topic live o dal primo frame di un .npy) e misura
la percettibilita', la stessa `compute_perturbation_magnitude` usata come
secondo obiettivo di NSGA-III, in tre modi:

  1. un operatore alla volta, con gli altri spenti (gene a -1), a intensita'
     media (gene 0) e massima (gene +1);
  2. ablazione sui genomi scelti da NSGA-III in un history.json: Chamfer del
     genoma completo e Chamfer togliendo un operatore per volta;
  3. il drift temporale, che si accumula, misurato dopo N frame consecutivi
     (default 12, cioe' una finestra da 1 m a 0.8 m/s e 10 Hz).

Attribuisce cosi' la Chamfer dei genomi scelti ai singoli operatori. Non
modifica nulla.

Uso:
    # con Isaac Sim in Play e add_intensity_node attivo
    python3 src/perturbations/chamfer_by_operator.py

    # con i genomi scelti in una run, anche sui soli punti entro 10 m
    python3 src/perturbations/chamfer_by_operator.py \
        --history data/attack/<run>/history.json --radius 10

    # da un file .npy (primo frame)
    python3 src/perturbations/chamfer_by_operator.py --npy data/frame_sequence.npy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.perturbations.perturbation_generator import PerturbationGenerator  # noqa: E402
from src.perturbations.profile_operators import grab_cloud_from_topic  # noqa: E402

# Nome leggibile -> indici dei geni che lo controllano. Stessa mappa di
# encode_perturbation; "noise" comprende direzione e intensita'.
OPERATORS = {
    "noise (dir 0-2, int 3, corr 11)": [0, 1, 2, 3, 11],
    "curvature_strength (4)":          [4],
    "dropout_rate (5)":                [5],
    "ghost_ratio (6)":                 [6],
    "cluster (dir 7-9, str 10)":       [7, 8, 9, 10],
    "geometric_distortion (12)":       [12],
    "edge_attack_strength (13)":       [13],
    "temporal_drift_strength (14)":    [14],
    "scanline_strength (15)":          [15],
    "strategic_ghost (16)":            [16],
}
DIRECTION_GENES = {0, 1, 2, 7, 8, 9}


def off_genome(n=17):
    """Tutti gli operatori spenti: gene a -1 -> parametro a 0."""
    return -np.ones(n)


def with_operator(genes, level, direction=(1.0, 0.0, 0.0)):
    """Genoma con il solo operatore `genes` acceso al livello `level`."""
    g = off_genome()
    for i in genes:
        g[i] = level
    # I geni direzione non sono intensita': si usa una direzione fissa.
    for base in (0, 7):
        if base in genes:
            g[base:base + 3] = direction
    # curvature_strength da solo non muove punti: si misura insieme al rumore.
    if genes == [4]:
        g[0:3] = direction
        g[3] = level
    return g


def perceptibility(gen, cloud, genome, frames=1, seed=0):
    """Percettibilita' (ultimo frame, media) su `frames` nuvole identiche in sequenza.

    Piu' frame servono solo al drift temporale, che si accumula; per gli altri
    operatori un frame basta. Seed fisso per confrontare a parita' di estrazione.
    """
    params = gen.encode_perturbation(np.asarray(genome, dtype=np.float64))
    gen.reset_temporal_state()
    vals = []
    for k in range(frames):
        out = gen.apply_perturbation(cloud, params, seed=seed + k)
        vals.append(gen.compute_perturbation_magnitude(cloud, out, params))
    return float(vals[-1]), float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/carter/lidar_with_intensity")
    ap.add_argument("--npy", default=None)
    ap.add_argument("--history", default=None,
                    help="history.json: ablazione sui genomi scelti")
    ap.add_argument("--frames", type=int, default=12,
                    help="frame consecutivi per far accumulare il drift")
    ap.add_argument("--radius", type=float, default=None,
                    help="se dato, misura anche la Chamfer sui soli punti "
                         "entro questo raggio dal sensore (m)")
    args = ap.parse_args()

    if args.npy:
        seq = np.load(args.npy, allow_pickle=True)
        cloud = np.asarray(seq[0], dtype=np.float64)
        if cloud.shape[1] == 3:
            cloud = np.hstack([cloud, np.full((len(cloud), 1), 100.0)])
    else:
        print(f"In attesa di una nuvola su {args.topic} ...")
        cloud = grab_cloud_from_topic(args.topic)
        if cloud is None:
            print("Nessuna nuvola ricevuta. Isaac Sim e' in Play? add_intensity gira?")
            return 1

    r = np.linalg.norm(cloud[:, :3], axis=1)
    print(f"Nuvola: {len(cloud)} punti   raggio mediano {np.median(r):.1f} m   "
          f"90° pct {np.percentile(r, 90):.1f} m   max {r.max():.1f} m\n")

    # Stessi limiti del perturbation_node.
    gen =PerturbationGenerator(max_point_shift=0.05, noise_std=0.02,
                                max_dropout_rate=0.15)

    clouds = {"tutta la nuvola": cloud}
    if args.radius:
        clouds[f"entro {args.radius:.0f} m"] = cloud[r <= args.radius]

    for cname, c in clouds.items():
        print(f"=== {cname} ({len(c)} punti) ===")
        print(f"{'operatore da solo':<36} {'gene 0':>10} {'gene +1':>10}   "
              f"{'(cm, ultimo frame)':>18}")
        print("-" * 80)
        base, _ = perceptibility(gen, c, off_genome(), frames=1)
        print(f"{'tutti spenti':<36} {base:>10.2f} {base:>10.2f}")
        for name, genes in OPERATORS.items():
            row = []
            for level in (0.0, 1.0):
                g = with_operator(genes, level)
                frames = args.frames if 14 in genes else 1
                last, _ = perceptibility(gen, c, g, frames=frames)
                row.append(last)
            note = f"  (dopo {args.frames} frame)" if 14 in genes else ""
            print(f"{name:<36} {row[0]:>10.2f} {row[1]:>10.2f}{note}")
        print()

    if args.history:
        with open(args.history) as f:
            H = json.load(f)
        print(f"=== ablazione sui genomi scelti in {args.history} ===")
        print("(Chamfer del genoma completo, poi con UN operatore spento per volta;\n"
              " la differenza e' quanto quell'operatore pesa sulla percettibilita')\n")
        for w in H:
            if not w.get("genome"):
                continue
            g = np.array(w["genome"], dtype=np.float64)
            full, _ = perceptibility(gen, cloud, g, frames=args.frames)
            print(f"finestra {w['window']}: riportata {w.get('perturbation', float('nan')):7.2f} cm   "
                  f"ricalcolata {full:7.2f} cm")
            for name, genes in OPERATORS.items():
                g2 = g.copy()
                for i in genes:
                    if i not in DIRECTION_GENES:
                        g2[i] = -1.0
                v, _ = perceptibility(gen, cloud, g2, frames=args.frames)
                print(f"    senza {name:<34} {v:7.2f} cm   delta {full - v:+7.2f}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
