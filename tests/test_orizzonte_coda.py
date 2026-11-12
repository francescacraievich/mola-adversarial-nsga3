"""Horizon ablation: ORIZZONTE_M in the campaign queue, dry runs.
Off (1) the queue is unchanged; on, it passes --horizon to the orchestrator, adds the suffix
_H<horizon> and checks FINESTRE * ORIZZONTE_M = 5 and the root name.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ORIGINALE = Path(os.environ.get("ORIZZONTE_ORIGINALE", os.path.expanduser("~/mola_lavoro/orizzonte_ablazione/repo_originale")))
AMBIENTE = Path(os.path.expanduser("~/mola_lavoro/coda_notte_semi_2001_3001/ambiente_coda.env"))


def _a_secco(radice_codice, cond, radice, **extra):
    env = {k: v for k, v in os.environ.items() if k not in ("ORIZZONTE_M", "FINESTRE", "A_SECCO")}
    for riga in AMBIENTE.read_text().splitlines():
        if "=" in riga:
            k, v = riga.split("=", 1)
            env[k] = v
    env.update(SEMI="2001", RADICE=str(radice), **extra)
    r = subprocess.run(["bash", "scripts/coda_serieB.sh", "--a-secco", cond], cwd=radice_codice, env=env,
                       capture_output=True, text=True, timeout=300)
    # the orchestrator line is printed with printf %q (spaces escaped by backslashes): normal form
    return (r.stdout + r.stderr).replace("\\ ", " ")


def _falliti(out):
    return [l for l in out.splitlines() if "CONTROLLO FALLITO" in l and "simulazione in corso" not in l]


def _norm(t, radice):
    t = t.replace(str(radice), "REPO")
    t = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "DATA", t)
    t = re.sub(r"md5 dei file: [0-9a-f]+", "md5", t)
    t = re.sub(r"battito vecchio di \d+ s|acceso da \d+ s|disco \d+ GB liberi \(\d+% usato\)|disco: \d+ GB liberi", "X", t)
    return t


def test_acceso_2p5(tmp_path):
    out = _a_secco(ROOT, "nsga3_senzabudget", tmp_path / "orizzonte_H2p5", ORIZZONTE_M="2.5", FINESTRE="2")
    assert not _falliti(out), _falliti(out)
    assert "--horizon 2.5 --drive-timeout 112.5" in out and "_H2p5_coda_N2_" in out and "--max-windows 2" in out


def test_acceso_5(tmp_path):
    out = _a_secco(ROOT, "casuale_senzabudget", tmp_path / "orizzonte_H5", ORIZZONTE_M="5", FINESTRE="1")
    assert not _falliti(out), _falliti(out)
    assert "--horizon 5 --drive-timeout 225" in out and "_H5_coda_N1_" in out


def test_finestre_incoerenti(tmp_path):
    out = _a_secco(ROOT, "nsga3_senzabudget", tmp_path / "orizzonte_H2p5", ORIZZONTE_M="2.5", FINESTRE="5")
    assert any("FINESTRE * ORIZZONTE_M = 5" in f for f in _falliti(out))


def test_radice_senza_nome(tmp_path):
    out = _a_secco(ROOT, "nsga3_senzabudget", tmp_path / "radice_qualunque", ORIZZONTE_M="2.5", FINESTRE="2")
    assert any("la radice deve contenere 'orizzonte'" in f for f in _falliti(out))


@pytest.mark.skipif(not (ORIGINALE / "scripts/coda_serieB.sh").exists(), reason="copia originale assente")
@pytest.mark.parametrize("cond", ["nsga3_senzabudget", "casuale_eps20", "massimo", "randpuro_eps20"])
def test_spento_identico(tmp_path, cond):
    uscite = [_norm(_a_secco(r, cond, tmp_path / "radice"), r) for r in (ORIGINALE, ROOT)]
    assert uscite[0] == uscite[1]
