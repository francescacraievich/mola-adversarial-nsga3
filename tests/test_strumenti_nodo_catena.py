"""STRUMENTI_NODO in run_campaign.sh: --strumenti-ritardo in the repetition folder.

node_start is extracted from the script and run with a fake python3 that writes its
arguments: off by default, once only if NODE_EXTRA already asks for it.
"""
import os
import re
import subprocess
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]
SCRIPT = RADICE / "scripts" / "run_campaign.sh"


def _node_start():
    testo = SCRIPT.read_text()
    m = re.search(r"^node_start\(\) \{.*?^\}$", testo, re.S | re.M)
    assert m, "node_start non trovata in run_campaign.sh"
    return m.group(0)


def _argomenti(tmp_path, strumenti, node_extra=""):
    finto = tmp_path / "bin"
    finto.mkdir(exist_ok=True)
    uscita = tmp_path / "argomenti.txt"
    (finto / "python3").write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > "{uscita}"\n')
    (finto / "python3").chmod(0o755)
    prog = "\n".join([
        f"OUT={tmp_path}/rep_01",
        "MIN_LATENCY_MS=140 MAX_DRIFT_LAG_S= PERC_TERMS=0 STRUCTURAL= NODE_SB_FLAGS= OPERATOR_NODE_FLAGS= SERIE_B=0",
        f"STRUMENTI_NODO={strumenti}",
        f"NODE_EXTRA='{node_extra}'",
        _node_start(),
        f'node_start "{tmp_path}/nodo.log" ""',
        'wait "$NODE_PID"',
    ])
    env = dict(os.environ, PATH=f"{finto}:{os.environ['PATH']}")
    subprocess.run(["bash", "-c", prog], check=True, env=env, timeout=30)
    return uscita.read_text().split("\n")


def test_spento_di_default(tmp_path):
    arg = _argomenti(tmp_path, 0)
    assert "--strumenti-ritardo" not in arg


def test_acceso_nella_cartella_della_ripetizione(tmp_path):
    arg = _argomenti(tmp_path, 1)
    i = arg.index("--strumenti-ritardo")
    assert arg[i + 1] == f"{tmp_path}/rep_01/ritardo_nodo.csv"
    assert arg.count("--strumenti-ritardo") == 1


def test_non_raddoppia_se_node_extra_lo_chiede(tmp_path):
    arg = _argomenti(tmp_path, 1, "--strumenti-ritardo /altrove/r.csv")
    assert arg.count("--strumenti-ritardo") == 1
    assert arg[arg.index("--strumenti-ritardo") + 1] == "/altrove/r.csv"


def test_scritto_in_ambiente():
    assert ('[[ "$STRUMENTI_NODO" != "1" ]] || echo "strumenti del ritardo del nodo: '
            'STRUMENTI_NODO=1, uscita in $OUT/ritardo_nodo*.csv" >> "$OUT/ambiente.txt"') in SCRIPT.read_text()


def test_default_zero_con_set_u():
    # without the variable the script (set -u) must see it defined and off
    testo = SCRIPT.read_text()
    assert re.search(r'^STRUMENTI_NODO="\$\{STRUMENTI_NODO:-0\}"', testo, re.M)
    assert re.search(r"^set -[a-z]*u", testo, re.M)
