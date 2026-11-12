"""Bench chains of scripts/prova_banco_serieB_{37c,38}_catena.sh.

Fake root (CATENA_ROOT) with a fake wrapper and fake command files (COMANDI_SERIEB): no
chain process, no ROS node.
"""
import os
import subprocess
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# fake wrapper: records the call and the K= line of the copy, writes outcome and log
INVOLUCRO_FINTO = r"""#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 9
secco=0; [[ "$1" == "--a-secco" ]] && { secco=1; shift; }
echo "$secco $2 $3 $(grep '^K=' "$1")" >> chiamate.txt
[[ $secco == 1 ]] && exit 0
OUT="data/attack/verifica/$(date +%F)_involucro_$2"; mkdir -p "$OUT"
rep="data/attack/verifica/prova/straight/braccio_$2/rep_01"; mkdir -p "$rep"
echo "== 12:00:00 straight / braccio_$2 / ripetizione 1 (tentativo 1)  ->  $rep" > "$OUT/comando.log"
v="ESITO_${2//[.-]/_}"; esito="${!v:-COMPLETATA}"
v="SENZA_COMPLETATA_${2//[.-]/_}"; [[ -n "${!v:-}" ]] || touch "$rep/completata"
echo "{\"esito\": \"$esito\"}" > "$OUT/prova_stato_finale.json"
v="RC_${2//[.-]/_}"; exit "${!v:-0}"
"""

CATENE = {
    "37c": ("3.7c_deriva_reset", "3.7c_deriva_keep"),
    "38": ("3.8a_massimo_spawn_r10", "3.8b_massimo_motion_r10"),
}


def _radice(tmp, catena):
    r = tmp / "radice"
    (r / "scripts").mkdir(parents=True)
    (r / "data" / "attack" / "verifica").mkdir(parents=True)
    (r / "scripts" / "prova_banco_serieB_campagna.sh").write_text(INVOLUCRO_FINTO)
    c = tmp / "comandi"
    c.mkdir()
    for n in CATENE[catena]:
        (c / f"2026-10-24_prova_{n}_REL.sh").write_text(
            f'#!/usr/bin/env bash\n# {n}\nK="K_RAMPA"                    # esito 3.6: 1.0 | 1.5\n'
            'case "$K" in 1.0|1.5) : ;; *) echo "segnaposto K_RAMPA non riempito: $K"; exit 2 ;; esac\n'
            'scripts/run_campaign.sh straight fixed 3 1 16 8\n')
    return r, c


def _lancia(tmp, catena, *args, **extra):
    r, c = (tmp / "radice", tmp / "comandi")
    env = dict(os.environ, CATENA_ROOT=str(r), COMANDI_SERIEB=str(c), VAULT_LOG_PREFISSO="prefisso ", **extra)
    return subprocess.run(["bash", str(ROOT / "scripts" / f"prova_banco_serieB_{catena}_catena.sh"), *args],
                          capture_output=True, text=True, env=env, timeout=60)


def _chiamate(tmp):
    f = tmp / "radice" / "chiamate.txt"
    return f.read_text().splitlines() if f.exists() else []


@pytest.mark.parametrize("catena", ["37c", "38"])
def test_k_non_valida(tmp_path, catena):
    _radice(tmp_path, catena)
    for args in ((), ("2.0",), ("--a-secco",), ("K_RAMPA",)):
        p = _lancia(tmp_path, catena, *args)
        assert p.returncode == 2 and "uso:" in p.stdout
    assert _chiamate(tmp_path) == [] and not (tmp_path / "comandi" / "catene").exists()


@pytest.mark.parametrize("catena,tmax", [("37c", "720"), ("38", "10800")])
def test_due_corse_con_k_sostituita(tmp_path, catena, tmax):
    r, c = _radice(tmp_path, catena)
    originali = {f: f.read_text() for f in c.glob("*.sh")}
    p = _lancia(tmp_path, catena, "1.5")
    assert p.returncode == 0, p.stdout + p.stderr
    a, b = CATENE[catena]
    ch = _chiamate(tmp_path)
    assert len(ch) == 2
    assert ch[0].startswith(f"0 {a}_k1.5 {tmax} K=\"1.5\"") and ch[1].startswith(f"0 {b}_k1.5 {tmax} K=\"1.5\"")
    # originals unchanged, new copies with a single different line
    assert {f: f.read_text() for f in c.glob("*.sh")} == originali
    copie = sorted((c / "catene").glob(f"*_{catena}_k1.5/*.sh"))
    assert [x.name for x in copie] == sorted(f"2026-10-24_prova_{n}_REL_k1.5.sh" for n in (a, b))
    for x in copie:
        o = c / x.name.replace("_k1.5.sh", ".sh")
        diff = [(u, v) for u, v in zip(o.read_text().splitlines(), x.read_text().splitlines()) if u != v]
        assert diff == [('K="K_RAMPA"                    # esito 3.6: 1.0 | 1.5', 'K="1.5"                    # esito 3.6: 1.0 | 1.5')]
    assert p.stdout.count("COMPLETATA, ripetizioni completate: 1") == 2


@pytest.mark.parametrize("catena", ["37c", "38"])
@pytest.mark.parametrize("guasto", ["ESITO", "SENZA_COMPLETATA", "RC"])
def test_si_ferma_alla_prima_corsa_non_completata(tmp_path, catena, guasto):
    _radice(tmp_path, catena)
    a, _ = CATENE[catena]
    chiave = f"{guasto}_{a}_k1.0".replace(".", "_").replace("-", "_")
    valore = {"ESITO": "INTERROTTA", "SENZA_COMPLETATA": "1", "RC": "3"}[guasto]
    p = _lancia(tmp_path, catena, "1.0", **{chiave: valore})
    assert p.returncode == (3 if guasto == "RC" else 1), p.stdout
    assert len(_chiamate(tmp_path)) == 1 and "catena fermata" in p.stdout


@pytest.mark.parametrize("catena", ["37c", "38"])
def test_controlli_prima_di_ogni_lancio(tmp_path, catena):
    r, c = _radice(tmp_path, catena)
    a, b = CATENE[catena]
    # wrapper folder of the second run already present
    inv = r / "data" / "attack" / "verifica" / f"{date.today():%Y-%m-%d}_involucro_{b}_k1.5"
    inv.mkdir()
    p = _lancia(tmp_path, catena, "1.5")
    assert p.returncode == 2 and "prima di ogni lancio" in p.stdout and _chiamate(tmp_path) == []
    inv.rmdir()
    # command file without the placeholder
    f = c / f"2026-10-24_prova_{b}_REL.sh"
    f.write_text(f.read_text().replace('K="K_RAMPA"', 'K="1.0"'))
    p = _lancia(tmp_path, catena, "1.5")
    assert p.returncode == 2 and "invece di una" in p.stderr and _chiamate(tmp_path) == []


@pytest.mark.parametrize("catena", ["37c", "38"])
def test_a_secco(tmp_path, catena):
    _radice(tmp_path, catena)
    p = _lancia(tmp_path, catena, "--a-secco", "1.0")
    assert p.returncode == 0, p.stdout + p.stderr
    a, b = CATENE[catena]
    ch = _chiamate(tmp_path)
    assert [x.split()[:2] for x in ch] == [["1", f"{a}_k1.0"], ["1", f"{b}_k1.0"]]
    assert all(x.endswith('K="1.0"                    # esito 3.6: 1.0 | 1.5') for x in ch)
    assert not (tmp_path / "comandi" / "catene").exists()          # copies in /tmp
    assert not list((tmp_path / "radice" / "data" / "attack" / "verifica").glob("*_involucro_*"))


def test_nessun_riferimento_vietato():
    for n in ("37c_catena", "38_catena", "33b_registrazione"):
        t = (ROOT / "scripts" / f"prova_banco_serieB_{n}.sh").read_text()
        # compound words split in pieces: they must not appear whole in the source
        for parola in ("Cla" + "ude", "Anth" + "ropic", "Co-" + "Authored", "." + "cla" + "ude"):
            assert parola not in t
