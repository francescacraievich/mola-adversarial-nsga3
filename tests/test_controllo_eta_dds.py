"""Cloud age check (--controllo-eta) and DDS profile (--dds-profile).

Synthetic data: fake MOLA DEBUG intake lines and fake per-frame node log.
"""

import datetime
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optimization import attack_orchestrator as ao  # noqa: E402

W0 = datetime.datetime(2026, 10, 29, 10, 0, 0).timestamp()
SOGLIA = 0.40


def _ora(t):
    d = datetime.datetime.fromtimestamp(t)
    return f"{d.hour:02d}:{d.minute:02d}:{d.second + d.microsecond / 1e6:07.4f}"


def _presa(t, stamp):
    return (f"[{_ora(t)}|DEBUG|Generator (raw)] Processing observation "
            f"type='mrpt::obs::CObservationPointCloud' label='lidar' stamp={stamp:.4f}")


def _frame_log(path, righe):
    """rows: (t_arr_wall, victim_stream, stamp, t_dds_src)."""
    with open(path, "w") as f:
        f.write("t_arr_wall,victim_stream,stamp_main_out,stamp_victim,t_dds_src\n")
        for t, vs, st, src in righe:
            sm, sv = (st, "") if vs == 0 else ("", st)
            f.write(f"{t},{vs},{sm},{sv},{src}\n")


def _tratto(n=20, ritardo=0.05, salta=(), vs=0):
    """n clouds at 10 Hz from the segment [W0, W0 + n/10]: log rows and MOLA intakes after
    `ritardo` s (no intake for the indices in `salta`)."""
    righe, log = [], []
    for i in range(n):
        t = W0 + 0.1 * i
        st = 10.0 + 0.1 * i
        righe.append((t + 0.01, vs, st, t))
        if i not in salta:
            log.append(_presa(t + (ritardo(i) if callable(ritardo) else ritardo), st))
    return righe, "\n".join(["altra riga"] + log) + "\n", W0, W0 + 0.1 * (n - 1)


def test_eta_alle_prese_e_ok(tmp_path):
    righe, log, w0, w1 = _tratto()
    fl = tmp_path / "frame_log.csv"
    _frame_log(fl, righe)
    m = ao.misura_eta(log, ao.righe_flusso(fl), w0, w1, SOGLIA)
    assert m["esito"] == "ok"
    assert m["n_prese"] == 20 and m["n_mai_prese"] == 0
    assert m["eta_max_s"] == pytest.approx(0.05, abs=1e-3)


def test_eta_oltre_soglia_e_guasto(tmp_path):
    righe, log, w0, w1 = _tratto(ritardo=lambda i: 0.9 if i == 7 else 0.05)
    fl = tmp_path / "frame_log.csv"
    _frame_log(fl, righe)
    m = ao.misura_eta(log, ao.righe_flusso(fl), w0, w1, SOGLIA)
    assert m["esito"] == "guasto"
    assert m["eta_max_s"] == pytest.approx(0.9, abs=1e-3)
    assert "eta' massima" in m["motivo"]


def test_mai_prese_e_guasto(tmp_path):
    # cloud 5 never taken: published before end - 0.40 s
    righe, log, w0, w1 = _tratto(salta=(5,))
    fl = tmp_path / "frame_log.csv"
    _frame_log(fl, righe)
    m = ao.misura_eta(log, ao.righe_flusso(fl), w0, w1, SOGLIA)
    assert m["esito"] == "guasto" and m["n_mai_prese"] == 1
    assert m["eta_mai_prese_max_s"] == pytest.approx(w1 - (W0 + 0.5), abs=1e-3)


def test_mai_presa_in_coda_non_conta(tmp_path):
    # the last cloud, published after end - 0.40 s, may not be taken yet
    righe, log, w0, w1 = _tratto(salta=(19,))
    fl = tmp_path / "frame_log.csv"
    _frame_log(fl, righe)
    assert ao.misura_eta(log, ao.righe_flusso(fl), w0, w1, SOGLIA)["esito"] == "ok"


def test_flusso_vittima_e_gemello_separati(tmp_path):
    # victim rows (victim_stream 1, stamp_victim) and twin rows with different stamps
    rg, log_g, w0, w1 = _tratto(vs=0)
    rv = [(t, 1, st + 100.0, src) for t, _, st, src in rg]
    fl = tmp_path / "frame_log.csv"
    _frame_log(fl, rg + rv)
    assert len(ao.righe_flusso(fl)) == 20 and len(ao.righe_flusso(fl, vittima=True)) == 20
    # the twin log takes none of the victim clouds: all never taken
    m = ao.misura_eta(log_g, ao.righe_flusso(fl, vittima=True), w0, w1, SOGLIA)
    assert m["esito"] == "guasto" and m["n_prese"] == 0


def test_senza_righe_debug_non_determinato(tmp_path):
    righe, _, w0, w1 = _tratto()
    fl = tmp_path / "frame_log.csv"
    _frame_log(fl, righe)
    m = ao.misura_eta("nessuna riga DEBUG\n", ao.righe_flusso(fl), w0, w1, SOGLIA)
    # no intake and clouds in the segment: never taken, hence failure
    assert m["esito"] == "guasto"
    m = ao.misura_eta("nessuna riga DEBUG\n", [], w0, w1, SOGLIA)
    assert m["esito"] == "non_determinato"


def test_salto_della_mezzanotte():
    w = datetime.datetime(2026, 10, 29, 23, 59, 59, 900000).timestamp()
    pr = ao.prese_mola(_presa(w + 0.2, 1.0), w)
    assert pr[0][0] == pytest.approx(w + 0.2, abs=1e-3)


def test_registri_nodo_correnti(tmp_path):
    import argparse
    for n in ("frame_log.csv", "frame_log_1.csv", "shadow_crc.csv", "shadow_crc_1.csv", "shadow_crc_2.csv"):
        (tmp_path / n).write_text("x\n")
    a = argparse.Namespace(frame_log_nodo=str(tmp_path / "frame_log.csv"),
                           shadow_crc_log=str(tmp_path / "shadow_crc.csv"))
    ao.registri_nodo_correnti(a)
    assert a.frame_log_nodo == str(tmp_path / "frame_log_1.csv")
    assert a.shadow_crc_log == str(tmp_path / "shadow_crc_2.csv")
    b = argparse.Namespace(frame_log_nodo=None, shadow_crc_log=None)
    ao.registri_nodo_correnti(b)
    assert b.frame_log_nodo is None and b.shadow_crc_log is None


def test_frame_log_corrente(tmp_path):
    p = tmp_path / "frame_log.csv"
    assert ao.frame_log_corrente(p) == str(p)
    p.write_text("x")
    (tmp_path / "frame_log_1.csv").write_text("x")
    assert ao.frame_log_corrente(p) == str(tmp_path / "frame_log_1.csv")


# --- rollout(): retry and exclusion ---------------------------------------

class _Mola:
    def __init__(self, path):
        self.log_path = path
        self.last_stop = None
        self.args = []


class _Node:
    fresh_cloud = False


def _prepara(monkeypatch, tmp_path, esiti):
    """Fake _rollout_once: each call writes the MOLA log of the attempt (failure or ok,
    from `esiti`) and returns a recognizable result."""
    righe, _, w0, w1 = _tratto()
    fl = tmp_path / "frame_log.csv"
    _frame_log(fl, righe)
    reg = tmp_path / "controllo_eta.jsonl"
    monkeypatch.setattr(ao, "CONTROLLO_ETA", {"soglia": SOGLIA, "frame_log": str(fl), "registro": str(reg), "finestre": {}})
    monkeypatch.setattr(ao, "_note_twin_log", lambda n, m: None)
    monkeypatch.setattr(ao, "_note_twin_fp", lambda n, m: None)
    chiamate = []

    def _once(genome, isaac, mola, node, goal, horizon, tag, **kw):
        i = len(chiamate)
        chiamate.append(tag)
        _, log, _, _ = _tratto(salta=(3, 4) if esiti[i] == "guasto" else ())
        mola.log_path = tmp_path / f"mola_{tag}.log"
        mola.log_path.write_text(log)
        node.last_drive_wall = (w0, w1)
        return (None, None, 0.0, "ok", 1.0, 10 + i)

    monkeypatch.setattr(ao, "_rollout_once", _once)
    return chiamate, reg


def test_gemello_guasto_rifatto_e_escluso(monkeypatch, tmp_path):
    chiamate, reg = _prepara(monkeypatch, tmp_path, ["guasto", "ok"])
    node, mola = _Node(), _Mola(None)
    res = ao.rollout(None, None, mola, node, [(1, 0)], 1.0, "w0_e3")
    assert chiamate == ["w0_e3", "w0_e3r"]
    # values from the second attempt, never from the first
    assert res[5] == 11
    assert [m["esito"] for m in node.last_eta_nuvola] == ["guasto", "ok"]
    assert [m["tentativo"] for m in node.last_eta_nuvola] == [1, 2]
    assert ao._twin_start_fields(node)["eta_nuvola"] == node.last_eta_nuvola
    voci = [json.loads(r) for r in reg.read_text().splitlines()]
    assert [v["tag"] for v in voci] == ["w0_e3", "w0_e3r"]
    assert {"eta_max_s", "n_prese", "n_mai_prese", "esito", "tentativo"} <= set(voci[0])


def test_gemello_secondo_guasto_escluso_ricerca_prosegue(monkeypatch, tmp_path):
    chiamate, _ = _prepara(monkeypatch, tmp_path, ["guasto", "guasto"])
    monkeypatch.setattr(ao, "STALL_POLICY", {**ao.STALL_POLICY, "window": 2})
    res = ao.rollout(None, None, _Mola(None), _Node(), [(1, 0)], 1.0, "w2_e3")
    assert chiamate == ["w2_e3", "w2_e3r"]
    # outcome outside VALID and COUNTABLE: never in fitness, archive or selection
    assert res[3] == ao.ESITO_ETA_RIPETUTO and res[2] == float("inf")
    assert res[3] not in ao.VALID and res[3] not in ao.COUNTABLE
    c = ao.CONTROLLO_ETA["finestre"]["2"]
    assert (c["tentativi"], c["rifatti"], c["ripetuti"]) == (2, 1, 1)
    assert c["quota_rifatti"] == 0.5


def test_riga_spezzata_da_ros_e_scarti(tmp_path):
    righe, log, w0, w1 = _tratto()
    linee = log.splitlines()
    # intake 4 split by a ROS line (TF_OLD_DATA), the rest after an empty line
    i = next(j for j, x in enumerate(linee) if "stamp=10.4000" in x)
    a, b = linee[i][:60], linee[i][60:]
    linee[i:i + 1] = [a + "[WARN] [1793195957.662695356] []: TF_OLD_DATA ignoring data", "", b]
    linee.append(f"[{_ora(W0 + 0.3)}|WARN |LidarOdometry] Dropping LiDAR scan (worker busy: "
                 "keeping only the freshest); dropped frames: 0.78%")
    fl = tmp_path / "frame_log.csv"
    _frame_log(fl, righe)
    m = ao.misura_eta("\n".join(linee), ao.righe_flusso(fl), w0, w1, SOGLIA)
    assert m["esito"] == "ok" and m["n_prese"] == 20 and m["n_drop_mola"] == 1


def test_vittima_guasto_ferma_la_run(monkeypatch, tmp_path):
    chiamate, _ = _prepara(monkeypatch, tmp_path, ["guasto", "ok"])
    # for the victim the stream is victim_stream 1: log with the victim rows
    righe, _, _, _ = _tratto(vs=1)
    _frame_log(tmp_path / "frame_log.csv", righe)
    vit = _Mola(None)

    def _once(genome, isaac, mola, node, goal, horizon, tag, **kw):
        chiamate.append(tag)
        _, log, w0, w1 = _tratto(salta=(2,))
        vit.log_path = tmp_path / "mola_victim.log"
        vit.log_path.write_text(log)
        node.last_drive_wall = (w0, w1)
        return (None, None, 0.0, "ok", 1.0, 5)

    monkeypatch.setattr(ao, "_rollout_once", _once)
    chiamate.clear()
    with pytest.raises(ao.GuastoEtaNuvola):
        ao.rollout(None, None, _Mola(None), _Node(), [(1, 0)], 1.0, "w0_applied", victim=vit)
    assert chiamate == ["w0_applied"]


def test_invarianza_senza_opzione(monkeypatch, tmp_path):
    monkeypatch.setattr(ao, "CONTROLLO_ETA", {"soglia": None, "frame_log": None, "registro": None})
    monkeypatch.setattr(ao, "controlla_eta_rollout", lambda *a, **k: pytest.fail("controllo acceso"))
    monkeypatch.setattr(ao, "_note_twin_log", lambda n, m: None)
    monkeypatch.setattr(ao, "_note_twin_fp", lambda n, m: None)
    monkeypatch.setattr(ao, "_rollout_once", lambda *a, **k: (None, None, 0.0, "ok", 1.0, 7))
    node = _Node()
    assert ao.rollout(None, None, _Mola(None), node, [(1, 0)], 1.0, "w0_e0")[5] == 7
    assert node.last_eta_nuvola is None
    assert "eta_nuvola" not in ao._twin_start_fields(node)


# --- DDS profile ------------------------------------------------------------

PROFILO = ROOT / "config" / "dds" / "fastdds_buffer4M_shm16M.xml"


def test_profilo_nella_repo():
    import hashlib
    assert hashlib.md5(PROFILO.read_bytes()).hexdigest().startswith("baadce5c")


def _spawn_env(monkeypatch, tmp_path, profilo):
    visto = {}

    class _P:
        def poll(self):
            return None

    def _popen(cmd, **kw):
        visto.update(kw["env"])
        return _P()

    monkeypatch.setattr(ao, "DDS_PROFILO", {"path": profilo, "md5": None})
    monkeypatch.setattr(ao.subprocess, "Popen", _popen)
    monkeypatch.setattr(ao.MolaRunner, "_count_processes", staticmethod(lambda: 0))
    monkeypatch.setattr(ao.MolaRunner, "_live", set())
    monkeypatch.setenv("FASTRTPS_DEFAULT_PROFILES_FILE", "/ereditato.xml")
    r = ao.MolaRunner("launch.sh", tmp_path)
    monkeypatch.setattr(r, "stop", lambda: None)
    r.spawn("w0_e0")
    r._f.close()
    return visto


def test_mola_senza_profilo_come_prima(monkeypatch, tmp_path):
    env = _spawn_env(monkeypatch, tmp_path, None)
    assert "FASTRTPS_DEFAULT_PROFILES_FILE" not in env


def test_mola_con_profilo(monkeypatch, tmp_path):
    env = _spawn_env(monkeypatch, tmp_path, str(PROFILO))
    assert env["FASTRTPS_DEFAULT_PROFILES_FILE"] == str(PROFILO)


def test_guardie_profilo(tmp_path):
    with pytest.raises(ValueError, match="non esiste"):
        ao.prepara_dds_profilo(tmp_path / "assente.xml", cwd=tmp_path)
    (tmp_path / "DEFAULT_FASTRTPS_PROFILES.xml").write_text("<x/>")
    with pytest.raises(ValueError, match="cartella di lavoro"):
        ao.prepara_dds_profilo(PROFILO, cwd=tmp_path)
    d = ao.prepara_dds_profilo(PROFILO, cwd=tmp_path / "..")
    assert d["path"] == str(PROFILO.resolve()) and d["md5"].startswith("baadce5c")


def test_main_rifiuta_profilo_assente(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["attack_orchestrator.py", "--dds-profile", str(tmp_path / "no.xml"),
                                      "--out", str(tmp_path / "run")])
    assert ao.main() == 2
    assert "non esiste" in capsys.readouterr().out


def test_main_controllo_eta_senza_registro(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["attack_orchestrator.py", "--controllo-eta", "0.4",
                                      "--out", str(tmp_path / "run")])
    assert ao.main() == 2
    assert "--frame-log-nodo" in capsys.readouterr().out


def _lib(tmp_path, dds, cwd):
    import subprocess
    cmd = (f'ROOT={ROOT}; OUT={tmp_path}/assente; A_SECCO=1; DDS_PROFILE="{dds}"; '
           f'source {ROOT}/scripts/prova_banco_serieB_lib.sh; echo "F=$FASTRTPS_DEFAULT_PROFILES_FILE"')
    return subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin"})


def test_lib_profilo(tmp_path):
    r = _lib(tmp_path, "", tmp_path)
    assert r.returncode == 0 and "F=\n" in r.stdout          # no profile: no variable
    r = _lib(tmp_path, PROFILO, tmp_path)
    assert r.returncode == 0 and f"F={PROFILO.resolve()}" in r.stdout
    assert _lib(tmp_path, tmp_path / "no.xml", tmp_path).returncode == 1
    (tmp_path / "DEFAULT_FASTRTPS_PROFILES.xml").write_text("<x/>")
    assert _lib(tmp_path, PROFILO, tmp_path).returncode == 1


def test_presa_dentro_la_riga_di_ros(tmp_path):
    # the MOLA line of intake 6 written inside the TF_OLD_DATA line, mid-word
    righe, log, w0, w1 = _tratto()
    linee = log.splitlines()
    i = next(j for j, x in enumerate(linee) if "stamp=10.6000" in x)
    presa = linee[i]
    linee[i:i + 1] = ["[WARN] [1793195957.662695356] []: TF_OLD_DATA ignoring data from the past "
                      "for frame base_link at time 28.700000 according to aut" + presa,
                      "hority Authority undetectable",
                      "Possible reasons are listed at http://wiki.ros.org/tf/Errors%20explained"]
    fl = tmp_path / "frame_log.csv"
    _frame_log(fl, righe)
    m = ao.misura_eta("\n".join(linee), ao.righe_flusso(fl), w0, w1, SOGLIA)
    assert m["esito"] == "ok" and m["n_prese"] == 20 and m["n_mai_prese"] == 0
    assert m["eta_max_s"] == pytest.approx(0.05, abs=1e-3)
