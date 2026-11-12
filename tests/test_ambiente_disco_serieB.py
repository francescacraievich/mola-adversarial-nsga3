"""Node environment check and disk report.

The versions of the interpreter that launches perturbation_node (python3 from PATH) are
written to ambiente.txt of every run; with SERIE_B=1 run_campaign.sh and the queue stop at
startup if they differ from config/ambiente_nodo_serieB.txt. Without SERIE_B
run_campaign.sh is unchanged. run_campaign.sh runs with fake python3, pgrep, ros2 and sleep
in PATH (harness from test_debug_mola_serieB): no real process.

The daily summary reports the free GB on the repo disk and the --lista estimate for the
remaining runs."""

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ORIGINALE = ROOT.parent / f"{ROOT.name}_originale"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import ambiente_nodo as an  # noqa: E402
import coda_serieB_stato as cs  # noqa: E402
from test_debug_mola_serieB import campagna  # noqa: E402

PY = shutil.which("python3")
RIF = ROOT / "config" / "ambiente_nodo_serieB.txt"


def _rif(path, **cambi):
    v, _ = an.versioni()
    v.update(cambi)
    path.write_text("# riferimento di prova\n" + "".join(f"{k} {x}\n" for k, x in v.items()))
    return path


def _controlla(rif, interprete=(PY,)):
    return subprocess.run([*interprete, str(ROOT / "scripts" / "ambiente_nodo.py"), "controlla",
                           "--riferimento", str(rif)], capture_output=True, text=True, timeout=120)


# ------------------------------------------------------------ ambiente_nodo.py
def test_controlla_uguale_e_diverso(tmp_path):
    r = _controlla(_rif(tmp_path / "uguale.txt"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "uguale al riferimento" in r.stdout
    r = _controlla(_rif(tmp_path / "diverso.txt", numpy="0.0.0"))
    assert r.returncode == 3
    assert "numpy: attesa 0.0.0, trovata" in r.stdout
    (tmp_path / "vuoto.txt").write_text("# niente\n")
    assert _controlla(tmp_path / "vuoto.txt").returncode == 2
    assert _controlla(tmp_path / "assente.txt").returncode == 2


def test_riferimento_della_repo_uguale_a_questa_macchina():
    """The fixed reference matches the node interpreter on this machine."""
    rif = an.leggi_riferimento(RIF)
    assert set(rif) == {"python", "numpy", "scipy", "rclpy"}
    r = _controlla(RIF)
    assert r.returncode == 0, r.stdout


def test_interprete_isolato_diverso():
    """python3 -I (system numpy and scipy) is a different environment: stop."""
    v_I = subprocess.run([PY, "-I", "-c", "import numpy; print(numpy.__version__)"],
                         capture_output=True, text=True, timeout=60).stdout.strip()
    if v_I == an.leggi_riferimento(RIF)["numpy"]:
        pytest.skip("con -I numpy ha la stessa versione: nessun ambiente diverso da provare")
    r = _controlla(RIF, (PY, "-I"))
    assert r.returncode == 3
    assert "diverso dal riferimento" in r.stdout


def test_versioni_con_pid(tmp_path):
    p = subprocess.Popen([PY, "-c", "import numpy, sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    try:
        for _ in range(100):
            if "numpy" in Path(f"/proc/{p.pid}/maps").read_text():
                break
            time.sleep(0.05)
        r = subprocess.run([PY, str(ROOT / "scripts" / "ambiente_nodo.py"), "versioni", "--riferimento",
                            str(_rif(tmp_path / "r.txt")), "--pid", str(p.pid)],
                           capture_output=True, text=True, timeout=120)
    finally:
        p.communicate(b"", timeout=30)
    v, dove = an.versioni()
    for k in ("python", "numpy", "scipy", "rclpy"):
        assert f"\n{k} {v[k]}" in "\n" + r.stdout
    assert "riferimento" in r.stdout and ": uguale" in r.stdout
    assert f"processo {p.pid}: numpy caricato da {Path(dove['numpy']).parent}" in r.stdout


# ------------------------------------------------------------ run_campaign.sh
def test_run_campaign_serie_b_ambiente_diverso_fermata(tmp_path):
    env = {"SERIE_B": "1", "SEEDS": "1001", "AMBIENTE_RIFERIMENTO": str(_rif(tmp_path / "d.txt", scipy="0.0.0"))}
    r, nodo, orch, _ = campagna(ROOT, tmp_path / "c", env, "nsga3")
    assert r.returncode == 1
    assert "ambiente del perturbation_node diverso dal riferimento: run fermata" in r.stdout
    assert "scipy: attesa 0.0.0" in r.stdout
    assert nodo is None and orch is None   # no node, no orchestrator
    assert "Isaac" not in r.stdout
    assert not list((tmp_path / "c").glob("*/camp/straight/*/rep_01"))


@pytest.mark.parametrize("arm", ["nsga3", "none"])
def test_run_campaign_serie_b_ambiente_uguale_e_ambiente_txt(tmp_path, arm):
    env = {"SERIE_B": "1", "SEEDS": "1001", "AMBIENTE_RIFERIMENTO": str(_rif(tmp_path / "u.txt"))}
    r, nodo, orch, _ = campagna(ROOT, tmp_path / "c", env, arm)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "uguale al riferimento" in r.stdout
    assert nodo is not None and orch is not None
    amb = list((tmp_path / "c").glob("*/camp/straight/*/rep_01/ambiente.txt"))
    assert len(amb) == 1
    testo = amb[0].read_text()
    v, _ = an.versioni()
    for k in ("python", "numpy", "scipy", "rclpy"):
        assert f"\n{k} {v[k]}" in "\n" + testo
    assert ": uguale" in testo
    assert "processo " in testo   # node interpreter (the fake node here)


def test_run_campaign_senza_serie_b_invariato(tmp_path):
    """Without SERIE_B: no check (even with a different reference) and no ambiente.txt;
    same output and arguments as the original copy."""
    env = {"AMBIENTE_RIFERIMENTO": str(_rif(tmp_path / "d.txt", numpy="0.0.0"))}
    r, nodo, orch, _ = campagna(ROOT, tmp_path / "n", env, "nsga3")
    assert r.returncode == 0, r.stdout[-3000:]
    assert "riferimento" not in r.stdout
    assert not list((tmp_path / "n").glob("*/camp/straight/*/rep_01/ambiente.txt"))
    if not ORIGINALE.is_dir():
        pytest.skip(f"copia originale assente: {ORIGINALE}")
    r0, nodo0, orch0, _ = campagna(ORIGINALE, tmp_path / "o", env, "nsga3")
    assert r0.returncode == r.returncode
    norm = lambda s, d: re.sub(r"\d\d:\d\d:\d\d", "<ora>", s.replace(str(d), "<T>"))
    assert norm(r.stdout, tmp_path / "n" / "c0") == norm(r0.stdout, tmp_path / "o" / "c0")
    assert [norm(x, tmp_path / "n" / "c0") for x in nodo] == [norm(x, tmp_path / "o" / "c0") for x in nodo0]
    assert [norm(x, tmp_path / "n" / "c0") for x in orch] == [norm(x, tmp_path / "o" / "c0") for x in orch0]


# ------------------------------------------------ ambiente.txt of bench runs and queue
def test_ambiente_su_file_con_versioni(tmp_path):
    cmd = (f'ROOT={ROOT}; OUT={tmp_path}; A_SECCO=0; source {ROOT}/scripts/prova_banco_serieB_lib.sh; '
           f'ambiente_su_file {tmp_path}/ambiente.txt')
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    testo = (tmp_path / "ambiente.txt").read_text()
    v, _ = an.versioni()
    assert "== interprete del perturbation_node" in testo
    for k in ("python", "numpy", "scipy", "rclpy"):
        assert f"\n{k} {v[k]}" in testo
    assert f"riferimento {RIF}: uguale" in testo


def test_coda_controlla_ambiente_prima_della_run():
    testo = (ROOT / "scripts" / "coda_serieB.sh").read_text()
    i = testo.index("esegui_run() {")
    j = testo.index('if [[ "$A_SECCO" != "1" && "$CONTROLLI_OK" != "1" ]]', i)
    assert "scripts/ambiente_nodo.py controlla" in testo[i:j]


# ------------------------------------------------------------ summary with disk space
def _stato(tmp_path):
    p = tmp_path / "stato.json"
    cs.scrivi(str(p), {"runs": {}})
    return p


def test_riepilogo_disco_simulato(tmp_path):
    runs = [("1001", "baseline"), ("1001", "nsga3_eps5"), ("2001", "casuale_eps5")]
    st = {"finestre": 5, "extra": 3, "pop": 32, "gen": 16, "topk": 2, "k_max": 64,
          "mb_rollout": 10.0, "mb_fisso": 500.0}
    disco = {"percorso": "/disco/finto", "gb_liberi": 123.45, "ora": "12:00:00", "margine_gb": 20,
             "fonte": "pilota finto", "stima": st}
    testo = cs.riepilogo({"runs": {}}, runs, oggi="2026-10-27", per_run=False, disco=disco)
    mb = [sum(cs.stima_run(c, ore={}, **st)["mb"][i] for _, c in runs) for i in (0, 1)]
    assert "Disco della repo (/disco/finto): 123.5 GB liberi" in testo
    assert f"run rimaste 3: servono circa {mb[0] / 1000:.1f} GB, al piu' {mb[1] / 1000:.1f} GB" in testo
    assert f"restano {123.45 - mb[1] / 1000 - 20:.1f} GB" in testo
    # little space: flagged
    disco["gb_liberi"] = 1.0
    assert "NON BASTA" in cs.riepilogo({"runs": {}}, runs, oggi="2026-10-27", per_run=False, disco=disco)
    # estimate not computable without a pilot run: free GB still reported
    disco["stima"] = dict(st, mb_rollout=None)
    t = cs.riepilogo({"runs": {}}, runs, oggi="2026-10-27", per_run=False, disco=disco)
    assert "1.0 GB liberi" in t and "stima per le 3 run rimaste non calcolabile" in t


def test_riepilogo_disco_reale_su_file(tmp_path):
    stato = _stato(tmp_path)
    r = subprocess.run([PY, "-I", str(ROOT / "scripts" / "coda_serieB_stato.py"), "riepilogo", "--disco", str(tmp_path),
                        "--margine-gb", "20", str(stato), str(tmp_path / "riep"), "1001:baseline"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    f = Path(r.stdout.strip())
    assert f.parent == tmp_path / "riep" and f.name.startswith("riepilogo_")
    testo = f.read_text()
    riga = next(x for x in testo.splitlines() if x.startswith("Disco della repo"))
    gb = float(riga.split("): ")[1].split(" GB")[0])
    assert abs(gb - shutil.disk_usage(tmp_path).free / 1e9) < 1.0
    assert "stima per le 1 run rimaste non calcolabile" in testo   # no pilot given


def test_riepilogo_senza_disco_invariato(tmp_path):
    runs = [("1001", "baseline")]
    testo = cs.riepilogo({"runs": {}}, runs, oggi="2026-10-27", per_run=False)
    assert "Disco della repo" not in testo
    if not ORIGINALE.is_dir():
        pytest.skip(f"copia originale assente: {ORIGINALE}")
    spec = importlib.util.spec_from_file_location("cs_orig", ORIGINALE / "scripts" / "coda_serieB_stato.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    norm = lambda s: s.splitlines()[1:]   # first line holds the write time
    assert norm(testo) == norm(m.riepilogo({"runs": {}}, runs, oggi="2026-10-27", per_run=False))


def test_coda_stato_con_disco(tmp_path):
    env = {k: v for k, v in os.environ.items()}
    env.update({"RADICE": str(tmp_path / "serieB")})
    r = subprocess.run(["bash", str(ROOT / "scripts" / "coda_serieB.sh"), "--stato", "baseline"],
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert f"Disco della repo ({ROOT})" in r.stdout
    assert "run rimaste" in r.stdout or "non calcolabile" in r.stdout
    assert not (tmp_path / "serieB").exists() or not list((tmp_path / "serieB").rglob("riepilogo_*"))
