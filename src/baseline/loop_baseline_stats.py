#!/usr/bin/env python3
"""
Statistiche di baseline su un insieme di run del percorso chiuso.

Legge, per ogni run N, data/trajectories/mola/run_N.tum, gt/run_N.tum e
gt/run_N_aligned.tum (riferimento gia' interpolato ai timestamp di MOLA, quindi
il confronto e' riga per riga senza sfasamenti temporali).

Per ogni run calcola:
  - ATE rispetto al riferimento;
  - chiusura del loop di MOLA e del riferimento, e differenza fra le due.

Fra le run calcola:
  - media e deviazione standard dell'ATE (fondo di rumore della baseline) e
    soglia media + 3 sigma oltre la quale un attacco conta come riuscito;
  - divergenza fra traiettorie MOLA e fra traiettorie di riferimento,
    accoppiate per ascissa curvilinea.

Servono entrambi i confronti: se MOLA e riferimento divergono nella stessa
misura fra due run, la variabilita' e' fisica (il robot percorre traiettorie
leggermente diverse) e non imputabile allo SLAM.

Uso:
    python3 src/baseline/loop_baseline_stats.py --runs 2 3 4 5 6 [--dir data/trajectories]
"""

import argparse
from pathlib import Path

import numpy as np


def load(path: Path):
    return np.loadtxt(path, comments="#")


def ate(mola_path: Path, gt_aligned_path: Path):
    """ATE 2D fra MOLA e riferimento interpolato agli stessi timestamp.

    Nessun allineamento di Umeyama: entrambe le traiettorie partono dalla stessa
    origine (MOLA si inizializza a 0,0,0 e il riferimento e' relativo alla posa
    iniziale), quindi l'errore e' direttamente confrontabile.
    """
    m = load(mola_path)
    g = load(gt_aligned_path)
    n = min(len(m), len(g))
    return np.linalg.norm(m[:n, 1:3] - g[:n, 1:3], axis=1)


def closure(path: Path):
    """Distanza fra ultima e prima posa: su un percorso chiuso e' il drift."""
    d = load(path)
    return float(np.linalg.norm(d[-1, 1:3] - d[0, 1:3]))


def pairwise(a_path: Path, b_path: Path, n_samples: int = 400):
    """Divergenza fra due traiettorie, accoppiate per ascissa curvilinea.

    Non per timestamp: se in una run il robot parte qualche secondo piu' tardi
    rispetto allo zero di MOLA, l'accoppiamento temporale confronta punti
    diversi dello stesso percorso e produce differenze dell'ordine dei metri su
    traiettorie geometricamente identiche. Accoppiando per frazione di percorso
    si confronta la geometria, che e' cio' che interessa per la ripetibilita'.

    Valido solo se le due traiettorie percorrono lo stesso tragitto, come qui
    dove ogni run esegue lo stesso rettangolo.
    """
    a, b = load(a_path), load(b_path)

    def resample(d):
        p = d[:, 1:3]
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] < 1e-6:
            return None
        u = np.linspace(0.0, s[-1], n_samples)
        return np.column_stack([np.interp(u, s, p[:, 0]),
                                np.interp(u, s, p[:, 1])])

    ra, rb = resample(a), resample(b)
    if ra is None or rb is None:
        return None
    return np.linalg.norm(ra - rb, axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, nargs="+", required=True)
    ap.add_argument("--dir", type=str, default="data/trajectories")
    args = ap.parse_args()

    base = Path(args.dir)
    runs, ates, closures = [], [], []

    print(f"\n{'='*72}")
    print("  PER RUN")
    print(f"{'='*72}")
    print(f"  {'run':>4}  {'pose':>5}  {'ATE med':>9}  {'ATE max':>9}  "
          f"{'chius.MOLA':>11}  {'chius.RIF':>10}  {'err':>7}")

    for r in args.runs:
        mp = base / "mola" / f"run_{r}.tum"
        ga = base / "gt" / f"run_{r}_aligned.tum"
        gp = base / "gt" / f"run_{r}.tum"
        if not (mp.exists() and ga.exists()):
            print(f"  {r:>4}  -- file mancanti, saltata")
            continue

        e = ate(mp, ga)
        cm, cg = closure(mp), closure(gp)
        runs.append(r)
        ates.append(e.mean())
        closures.append(cm)

        print(f"  {r:>4}  {len(load(mp)):>5}  "
              f"{e.mean()*1000:>7.1f}mm  {e.max()*1000:>7.1f}mm  "
              f"{cm*100:>9.2f}cm  {cg*100:>8.2f}cm  {abs(cm-cg)*100:>5.2f}cm")

    if len(runs) < 2:
        print("\n  Servono almeno due run valide.")
        return

    a = np.array(ates)
    c = np.array(closures)

    print(f"\n{'='*72}")
    print("  AGGREGATO")
    print(f"{'='*72}")
    print(f"  ATE medio           : {a.mean()*1000:.1f} mm")
    print(f"  deviazione standard : {a.std(ddof=1)*1000:.1f} mm")
    print(f"  escursione          : {a.min()*1000:.1f} - {a.max()*1000:.1f} mm")
    print(f"  chiusura media      : {c.mean()*100:.2f} cm  "
          f"(sigma {c.std(ddof=1)*100:.2f} cm)")

    # Soglia di significativita': un attacco conta come riuscito quando supera
    # la banda di rumore della baseline (media + 3 sigma).
    thr = a.mean() + 3 * a.std(ddof=1)
    print(f"\n  SOGLIA ATTACCO (media + 3 sigma): {thr*1000:.1f} mm")
    print(f"  un ATE sopra questo valore non e' spiegabile come rumore di baseline")

    print(f"\n{'='*72}")
    print("  DIVERGENZA FRA RUN  (accoppiata per ascissa curvilinea)")
    print(f"{'='*72}")
    print("  Se MOLA e riferimento divergono in misura simile, la variabilita'")
    print("  e' fisica e non imputabile allo SLAM.\n")
    print(f"  {'coppia':>10}  {'MOLA med':>9}  {'MOLA max':>9}  "
          f"{'RIF med':>9}  {'RIF max':>9}  {'eccesso':>8}")

    excesses = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            ri, rj = runs[i], runs[j]
            dm = pairwise(base / "mola" / f"run_{ri}.tum",
                          base / "mola" / f"run_{rj}.tum")
            dg = pairwise(base / "gt" / f"run_{ri}.tum",
                          base / "gt" / f"run_{rj}.tum")
            if dm is None or dg is None:
                continue
            exc = dm.mean() - dg.mean()
            excesses.append(exc)
            print(f"  {ri:>4}-{rj:<5}  {dm.mean()*1000:>7.1f}mm  {dm.max()*1000:>7.1f}mm  "
                  f"{dg.mean()*1000:>7.1f}mm  {dg.max()*1000:>7.1f}mm  "
                  f"{exc*1000:>6.1f}mm")

    if excesses:
        ex = np.array(excesses)
        print(f"\n  Eccesso medio di MOLA sul riferimento: {ex.mean()*1000:.1f} mm")
        print("  Questa e' la parte di variabilita' attribuibile allo SLAM;")
        print("  il resto e' il robot che percorre traiettorie leggermente diverse.")
    print()


if __name__ == "__main__":
    main()
