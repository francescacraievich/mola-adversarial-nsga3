#!/usr/bin/env python3
"""Library versions of the interpreter that runs the perturbation_node.

Run it with the same command as the node in run_campaign.sh (python3 from PATH,
without -I, ROS environment sourced): the versions reported are those of the
interpreter running this file. With -I, numpy and scipy would resolve to the
system packages (1.26.4 / 1.11.4) instead of the user site-packages used by the node.

  python3 scripts/ambiente_nodo.py versioni [--riferimento FILE] [--pid PID]
  python3 scripts/ambiente_nodo.py controlla [--riferimento FILE]

versioni: "key value" lines (python, numpy, scipy, rclpy) with interpreter and
module paths; with --riferimento also the comparison result; with --pid the
interpreter of that process and the numpy/scipy directories in its memory maps.
controlla: exit 0 if the versions match the reference, 3 if not,
2 if the reference cannot be read.
"""
import argparse
import importlib
import importlib.metadata
import os
import platform
import sys

RIFERIMENTO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                            "config", "ambiente_nodo_serieB.txt"))
MODULI = ("numpy", "scipy", "rclpy")   # libraries imported by the node and by src/perturbations


def versioni():
    """Return {key: version} and {key: path} for the current interpreter."""
    v = {"python": platform.python_version()}
    dove = {"python": os.path.realpath(sys.executable)}
    for m in MODULI:
        try:
            mod = importlib.import_module(m)
        except Exception as e:   # missing or unloadable module: the node would not start
            v[m] = "assente"
            dove[m] = f"{type(e).__name__}: {e}"
            continue
        ver = getattr(mod, "__version__", None)
        if ver is None:
            try:
                ver = importlib.metadata.version(m)
            except importlib.metadata.PackageNotFoundError:
                ver = "sconosciuta"
        v[m] = str(ver)
        dove[m] = os.path.dirname(os.path.realpath(getattr(mod, "__file__", "") or ""))
    return v, dove


def leggi_riferimento(path):
    rif = {}
    with open(path) as f:
        for riga in f:
            riga = riga.split("#", 1)[0].strip()
            if not riga:
                continue
            k, _, val = riga.partition(" ")
            if not val.strip():
                raise ValueError(f"riga senza versione: {riga!r}")
            rif[k] = val.strip()
    if not rif:
        raise ValueError("nessuna versione")
    return rif


def differenze(v, rif):
    return [f"{k}: attesa {rif[k]}, trovata {v.get(k, 'assente')}" for k in rif if v.get(k) != rif[k]]


def mappe_processo(pid):
    """Interpreter of the process and numpy/scipy directories among its loaded libraries."""
    righe = []
    try:
        righe.append(f"processo {pid}: interprete {os.readlink(f'/proc/{pid}/exe')}")
    except OSError as e:
        return [f"processo {pid}: non leggibile ({e.strerror})"]
    cartelle = {m: set() for m in ("numpy", "scipy")}
    try:
        with open(f"/proc/{pid}/maps") as f:
            for r in f:
                p = r.split()[-1] if len(r.split()) >= 6 else ""
                for m in cartelle:
                    i = p.find(f"/{m}/")
                    if i > 0:
                        cartelle[m].add(p[:i])
    except OSError as e:
        righe.append(f"processo {pid}: mappe non leggibili ({e.strerror})")
        return righe
    for m, c in cartelle.items():
        righe.append(f"processo {pid}: {m} caricato da {', '.join(sorted(c)) if c else 'non ancora caricato'}")
    return righe


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modo", choices=("versioni", "controlla"))
    ap.add_argument("--riferimento", default=None, help=f"file di riferimento (default {RIFERIMENTO})")
    ap.add_argument("--pid", type=int, help="con versioni: processo del nodo da descrivere")
    a = ap.parse_args(argv)
    path_rif = a.riferimento or RIFERIMENTO
    v, dove = versioni()

    if a.modo == "controlla":
        try:
            rif = leggi_riferimento(path_rif)
        except (OSError, ValueError) as e:
            print(f"riferimento dell'ambiente non leggibile: {path_rif} ({e})")
            return 2
        diff = differenze(v, rif)
        if diff:
            print(f"ambiente del perturbation_node diverso dal riferimento {path_rif}:")
            for d in diff:
                print(f"  {d}")
            print(f"  interprete {dove['python']}")
            return 3
        print(f"ambiente del perturbation_node uguale al riferimento {path_rif}: "
              + ", ".join(f"{k} {v[k]}" for k in rif))
        return 0

    print(f"interprete: {dove['python']} (python3 dal PATH, come il nodo)")
    for k in ("python",) + MODULI:
        print(f"{k} {v[k]}" + (f"  ({dove[k]})" if k != "python" else ""))
    print(f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}")
    print(f"PYTHONNOUSERSITE={os.environ.get('PYTHONNOUSERSITE', '')}")
    if a.riferimento is not None or os.path.exists(path_rif):
        try:
            diff = differenze(v, leggi_riferimento(path_rif))
            print(f"riferimento {path_rif}: " + ("uguale" if not diff else "DIVERSO - " + "; ".join(diff)))
        except (OSError, ValueError) as e:
            print(f"riferimento {path_rif}: non leggibile ({e})")
    if a.pid:
        for r in mappe_processo(a.pid):
            print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
