#!/usr/bin/env python3
"""Acceptance evaluation of a run (victim vs twin, metre by metre).

Read-only on the run; writes only into the new --out directory.

Usage: python3 analisi_accettazione.py <run (seed) dir> --out <new dir>
"""
import argparse, csv, glob, json, math, os, re, sys
import numpy as np

# acceptance thresholds
SOGLIA_R1_CM = 3.0
SOGLIA_R1_INFO_CM = 4.0          # informative only
SOGLIA_E1_CM = 3.0
SOGLIA_R2_ALL_CM = 5.0           # R2 reported: same stamp, motion-aligned
SOGLIA_R2_GRAD_CM = 3.0          # R2 step and interpolated: reported at 3 cm
SOGLIA_E2_CM = 3.0
SOGLIA_E3_STRADA_CM = 1.0
SOGLIA_E3_V_MS = 0.05
SOGLIA_R4_XY_M = 1e-6
SOGLIA_ETA_S = 0.40
SOGLIA_RIFACIMENTI = 0.05
N_METRI = 5

PASSA, NON_PASSA, ND = "PASSA", "NON PASSA", "non disponibile"


def virg(x, n=2, segno=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return ND
    s = f"{x:+.{n}f}" if segno else f"{x:.{n}f}"
    return s.replace(".", ",")


def k10(s):
    return int(round(float(s) * 10))


# ---------------------------------------------------------------- traces
def leggi_traccia(path):
    with open(path) as f:
        r = csv.reader(f)
        head = next(r)
        rows = [x for x in r if x and x[0] != head[0]]
    out = {}
    for j, c in enumerate(head):
        col = []
        for x in rows:
            v = x[j] if j < len(x) else ""
            try:
                col.append(float(v))
            except ValueError:
                col.append(v)
        out[c] = np.array(col, dtype=float) if all(isinstance(v, float) for v in col) else col
    return out


# ---------------------------------------------------------------- MOLA log
RE_WARN = re.compile(r"\[WARN\] \[[0-9.]+\] \[[^\]]*\]: TF_OLD_DATA[^\n]*\n(?:Possible reasons are listed at[^\n]*\n)?")
RE_ICP = re.compile(r"ICP \(kind=[01]\): goodness=[^\n]*?pose=\[([-0-9.eE+ ]+)\]")
RE_PATH = re.compile(r"\[LidarOdometry\] pathStep=\d+ timestamp=(\d{4})/(\d\d)/(\d\d),(\d\d):(\d\d):([0-9.]+)[^\n]*?isGood=(\d)")


def leggi_log_mola(path):
    """stamp (tenths of s) -> (x, y, yaw_deg) for pathStep lines with isGood=1; the pose is the
    last ICP line (kind=0 or kind=1) before the pathStep. TF_OLD_DATA warnings (with their
    'Possible reasons...' line) are stripped first, so lines split mid-word are rejoined."""
    txt = open(path, errors="replace").read()
    n_warn = len(RE_WARN.findall(txt))
    txt = RE_WARN.sub("", txt)
    ev = [(m.start(), "icp", m) for m in RE_ICP.finditer(txt)] + \
         [(m.start(), "path", m) for m in RE_PATH.finditer(txt)]
    ev.sort(key=lambda e: e[0])
    pose, tutte, senza_posa, last = {}, set(), 0, None
    for _, tipo, m in ev:
        if tipo == "icp":
            v = [float(a) for a in m.group(1).split()]
            last = (v[0], v[1], v[3])
            continue
        g = m.groups()
        giorno = int(g[2]) - 1
        st = giorno * 86400 + int(g[3]) * 3600 + int(g[4]) * 60 + float(g[5])
        k = k10(st)
        tutte.add(k)
        if g[6] == "1":
            if last is None:
                senza_posa += 1
            else:
                pose[k] = last
        last = None
    return {"pose": pose, "elaborate": tutte, "warn_tf": n_warn, "icp_senza_posa": senza_posa}


# ---------------------------------------------------------------- frame_log
def leggi_frame_log(path):
    col = {"victim_stream": [], "stamp_victim": [], "stamp_main_out": [], "t_pub_end_wall": [], "t_dds_src": []}
    with open(path) as f:
        r = csv.DictReader(f)
        for x in r:
            for c in col:
                v = x.get(c, "")
                try:
                    col[c].append(float(v))
                except (TypeError, ValueError):
                    col[c].append(float("nan"))
    return {c: np.array(v) for c, v in col.items()}


def nuvole_lato(fl, vittima, w0, w1, margine=3.0):
    """Clouds of one stream (victim: victim_stream 1, stamp_victim; twin: victim_stream 0,
    stamp_main_out) published in [w0 - margin, w1]."""
    if vittima:
        m = fl["victim_stream"] == 1
        st = fl["stamp_victim"]
    else:
        m = fl["victim_stream"] == 0
        st = fl["stamp_main_out"]
    tp = fl["t_pub_end_wall"]
    sel = m & np.isfinite(st) & np.isfinite(tp) & (tp >= w0 - margine) & (tp <= w1)
    idx = np.where(sel)[0]
    out = {}
    for i in idx:
        out[k10(st[i])] = (tp[i], fl["t_dds_src"][i])   # stamp -> (t_pub, t_dds_src)
    prima = [k for k, (t, _) in out.items() if t <= w0]
    stamp_w0 = max(prima, key=lambda k: out[k][0]) if prima else None
    return out, stamp_w0


# ---------------------------------------------------------------- side (victim or winner)
def prepara_lato(tr, log, fl, vittima):
    wall, t = tr["wall"], tr["t"]
    w0, w1 = float(wall[0] - t[0]), float(wall[-1])
    nuv, st_w0 = nuvole_lato(fl, vittima, w0, w1)
    pose = {k: log["pose"][k] for k in nuv if k in log["pose"]}
    nel = {k for k in nuv if nuv[k][0] >= w0}
    T = {}
    for k in pose:
        tsrc = nuv[k][1]
        T[k] = (float(np.interp(tsrc, wall, tr["true_x"])), float(np.interp(tsrc, wall, tr["true_y"])),
                float(np.interp(tsrc, wall, tr["travelled"])))
    cmd = np.asarray(tr["cmd_v"], float)
    mossa = np.where(cmd > 0)[0]
    i_stop = None
    if len(mossa):
        dopo = np.where((cmd == 0) & (np.arange(len(cmd)) > mossa[0]))[0]
        i_stop = int(dopo[0]) if len(dopo) else None
    lato = {"tr": tr, "w0": w0, "w1": w1, "nuvole": nuv, "stamp_w0": st_w0, "pose": pose,
            "nel_metro": nel, "T": T, "i_stop": i_stop, "n_pubblicate": len(nel),
            "n_non_elaborate": len([k for k in nel if k not in log["elaborate"]])}
    associa_valore(lato)
    return lato


def associa_valore(L):
    """Stamp of the pose held by the controller on each row, matched by value:
    est_dist = |p(row stamp) - p(first row stamp)| (frame independent). The reference stamp
    with the smallest residual is chosen; the held stamp never decreases and its cloud left
    Isaac before the row."""
    tr, pose, nuv = L["tr"], L["pose"], L["nuvole"]
    keys = sorted(pose)
    est, wall = np.asarray(tr["est_dist"], float), np.asarray(tr["wall"], float)
    cand = [k for k in keys if nuv[k][1] <= L["w0"] + 0.05][-8:]
    best = None
    for r in cand:
        pr = np.array(pose[r][:2])
        D = {k: float(np.hypot(*(np.array(pose[k][:2]) - pr))) for k in keys if k >= r}
        ks = sorted(D)
        held, err, cur = [], [], r
        for i in range(len(est)):
            ok = [k for k in ks if k >= cur and nuv[k][1] <= wall[i]]
            if not ok:
                held.append(cur); err.append(abs(D[cur] - est[i])); continue
            vic = [k for k in ok if abs(D[k] - est[i]) <= 5e-4]
            k = vic[0] if vic else min(ok, key=lambda q: abs(D[q] - est[i]))
            cur = k
            held.append(k); err.append(abs(D[k] - est[i]))
        sc = float(np.median(err)) if err else float("inf")
        if best is None or sc < best[0]:
            best = (sc, r, held, err, D)
    if best is None:
        L.update(rif=None, held=None, ass_err_med_cm=None, ass_err_max_cm=None, D=None)
        return
    sc, r, held, err, D = best
    L.update(rif=r, held=held, D=D, ass_err_med_cm=100 * float(np.median(err)),
             ass_err_max_cm=100 * float(np.max(err)))


# ---------------------------------------------------------------- per-metre measures
def errore_diff(v, w, kv, kw, psi):
    pv, pw = np.array(v["pose"][kv][:2]), np.array(w["pose"][kw][:2])
    Tv, Tw = np.array(v["T"][kv][:2]), np.array(w["T"][kw][:2])
    c, s = math.cos(psi), math.sin(psi)
    dT = Tv - Tw
    return float(np.hypot(*((pv - pw) - np.array([c * dT[0] - s * dT[1], s * dT[0] + c * dT[1]]))))


def r2_gradino(tv, tw):
    smax = min(tv["travelled"][-1], tw["travelled"][-1])
    g = np.arange(0, int(math.floor(smax * 100 + 1e-9)) + 1) / 100.0
    def d(tr, s):
        tt, ee = np.asarray(tr["travelled"]), np.asarray(tr["est_dist"])
        i = np.searchsorted(tt, s + 1e-12, side="right") - 1
        return ee[i] if i >= 0 else 0.0
    # travelled is non-decreasing: the found index is the last row with travelled <= s
    dv = np.array([d(tv, s) for s in g]); dw = np.array([d(tw, s) for s in g])
    return 100 * float(np.max(np.abs(dv - dw))), len(g)


def r2_interpolato(v, w):
    """MOLA pose placed on the true path at the t_dds_src of its cloud and interpolated
    between stamps; d = |p - p(controller reference)|."""
    def curva(L):
        if L["rif"] is None:
            return None
        ks = sorted(k for k in L["pose"] if k >= L["rif"] and L["nuvole"][k][1] <= L["w1"])
        s = np.array([L["T"][k][2] for k in ks]); d = np.array([L["D"][k] for k in ks])
        s = np.maximum.accumulate(s)
        u, idx = np.unique(s[::-1], return_index=True)
        return u, d[::-1][idx]
    cv, cw = curva(v), curva(w)
    if cv is None or cw is None:
        return None
    smax = min(v["tr"]["travelled"][-1], w["tr"]["travelled"][-1], cv[0][-1], cw[0][-1])
    g = np.arange(0, int(math.floor(smax * 100 + 1e-9)) + 1) / 100.0
    return 100 * float(np.max(np.abs(np.interp(g, *cv) - np.interp(g, *cw))))


def misura_metro(v, w):
    tv, tw = v["tr"], w["tr"]
    m = {}
    sf_v, sf_w = float(tv["travelled"][-1]), float(tw["travelled"][-1])
    m["s_fine_vittima_m"], m["s_fine_vincitore_m"] = sf_v, sf_w
    m["R1_cm"] = 100 * (sf_v - sf_w)
    for nome, L in (("vittima", v), ("vincitore", w)):
        i = L["i_stop"]; tr = L["tr"]
        m[f"s_stop_{nome}_m"] = float(tr["travelled"][i]) if i is not None else None
        m[f"d_stop_{nome}_m"] = float(tr["est_dist"][i]) if i is not None else None
        m[f"v_vera_stop_{nome}_ms"] = float(math.hypot(tr["true_vx"][i], tr["true_vy"][i])) if i is not None else None
        m[f"strada_dopo_stop_{nome}_cm"] = 100 * (float(tr["travelled"][-1]) - float(tr["travelled"][i])) if i is not None else None
        m[f"t_stop_{nome}_s"] = float(tr["t"][i]) if i is not None else None
        m[f"stamp_w0_{nome}"] = L["stamp_w0"] / 10 if L["stamp_w0"] is not None else None
        m[f"pubblicate_{nome}"] = L["n_pubblicate"]
        m[f"non_elaborate_{nome}"] = L["n_non_elaborate"]
        m[f"assoc_valore_med_cm_{nome}"] = L["ass_err_med_cm"]
        m[f"assoc_valore_max_cm_{nome}"] = L["ass_err_max_cm"]
        held = L["held"][i] if (L["held"] is not None and i is not None) else None
        m[f"stamp_posa_allo_stop_{nome}"] = held / 10 if held is not None else None
    # E2: three fixed points (slow-down, last command, stop)
    def tre(tr):
        c = np.asarray(tr["cmd_v"]); e = np.asarray(tr["est_dist"]); crociera = c[0]
        a = np.where(c < crociera - 1e-9)[0]; b = np.where(c != 0)[0]
        i_stop = [i for i in range(len(c)) if c[i] == 0 and len(b) and i > b[0]]
        return [float(e[a[0]]) if len(a) else None, float(e[b[-1]]) if len(b) else None,
                float(e[i_stop[0]]) if i_stop else None]
    e2v, e2w = tre(tv), tre(tw)
    m["E2_cm"] = [100 * (a - b) if a is not None and b is not None else None for a, b in zip(e2v, e2w)]
    # R2 step
    m["R2_gradino_cm"], m["R2_gradino_n_griglia"] = r2_gradino(tv, tw)
    # psi from the victim: MOLA displacement direction vs true one within the metre
    kv = sorted(k for k in v["pose"] if k in v["nel_metro"])
    psi = None
    if len(kv) >= 2:
        dp = np.array(v["pose"][kv[-1]][:2]) - np.array(v["pose"][kv[0]][:2])
        dT = np.array(v["T"][kv[-1]][:2]) - np.array(v["T"][kv[0]][:2])
        if np.hypot(*dp) > 0.2 and np.hypot(*dT) > 0.2:
            psi = math.atan2(dp[1], dp[0]) - math.atan2(dT[1], dT[0])
    m["psi_deg"] = math.degrees(psi) if psi is not None else None
    # same stamp and motion-aligned
    for var in ("stessa_stamp", "allineata"):
        if psi is None:
            m[f"R2_{var}_cm"] = m[f"E1_{var}_cm"] = None; m[f"n_comuni_{var}"] = 0; continue
        if var == "allineata":
            if v["stamp_w0"] is None or w["stamp_w0"] is None:
                m[f"R2_{var}_cm"] = m[f"E1_{var}_cm"] = None; m[f"n_comuni_{var}"] = 0; continue
            sh = w["stamp_w0"] - v["stamp_w0"]
        else:
            sh = 0
        mw = {k - sh: k for k in w["pose"] if k in w["nel_metro"]}
        com = sorted(k for k in kv if k in mw)
        m[f"n_comuni_{var}"] = len(com)
        m[f"spostamento_orologio_{var}_s"] = sh / 10
        if not com:
            m[f"R2_{var}_cm"] = m[f"E1_{var}_cm"] = None; continue
        ed = [100 * errore_diff(v, w, k, mw[k], psi) for k in com]
        m[f"R2_{var}_cm"] = float(max(ed))
        m[f"serie_{var}"] = [[k / 10, e] for k, e in zip(com, ed)]
        hv = m["stamp_posa_allo_stop_vittima"]; hw = m["stamp_posa_allo_stop_vincitore"]
        if hv is None or hw is None:
            m[f"E1_{var}_cm"] = None; continue
        lim = min(k10(hv), k10(hw) - sh)
        ok = [k for k in com if k <= lim]
        m[f"E1_{var}_cm"] = ed[com.index(ok[-1])] if ok else None
        m[f"E1_{var}_stamp_vittima"] = ok[-1] / 10 if ok else None
    m["R2_interpolato_cm"] = r2_interpolato(v, w)
    return m


# ---------------------------------------------------------------- shadow (R4)
def lati_scan(r):
    el, po = r.get("elaborata") or "", r.get("posa") or ""
    return (el in ("entrambe", "solo_vittima") or po in ("entrambe", "solo_vittima"),
            el in ("entrambe", "solo_ombra") or po in ("entrambe", "solo_ombra"))


def ombra_per_metro(rep, history):
    sc_json = os.path.join(rep, "shadow_check.json")
    sc_csv = os.path.join(rep, "shadow_check_scans.csv")
    confini = []
    ap = [h.get("victim_window", {}).get("open_stamp") for h in history]
    for i in range(len(history)):
        confini.append((None if i == 0 else ap[i], ap[i + 1] if i + 1 < len(history) else None))
    fonte, campo = None, None
    if os.path.isfile(sc_json):
        try:
            campo = json.load(open(sc_json)).get("identita_per_metro")
        except (OSError, ValueError):
            campo = None
    righe = []
    if os.path.isfile(sc_csv):
        righe = list(csv.DictReader(open(sc_csv)))
    out = []
    S = []
    for r in righe:
        try:
            k = k10(r["stamp_s"])
        except (KeyError, ValueError, TypeError):
            continue
        a, b = lati_scan(r)
        if a or b:
            S.append((k, a, b, r))
    kv = [k for k, a, _, _ in S if a]; ko = [k for k, _, b, _ in S if b]
    lim = min(max(kv), max(ko)) if kv and ko else None
    for i, (da, a) in enumerate(confini):
        dentro = [x for x in S if (da is None or x[0] > k10(da)) and (a is None or x[0] <= k10(a))
                  and lim is not None and x[0] <= lim]
        sv = [x[0] / 10 for x in dentro if x[1] and not x[2]]
        so = [x[0] / 10 for x in dentro if x[2] and not x[1]]
        dxy = []
        for x in dentro:
            if x[1] and x[2]:
                try:
                    dxy.append(float(x[3]["dxy_m"]))
                except (KeyError, ValueError, TypeError):
                    pass
        mx = max(dxy) if dxy else None
        if not dentro:
            es = "non_misurata"
        else:
            es = "ombra_divergente" if (sv or so or (mx is not None and mx > SOGLIA_R4_XY_M)) else "identica"
        d = {"finestra": i, "da_s": da, "a_s": a, "solo_vittima": sv, "solo_ombra": so,
             "max_dxy_m": mx, "n_stamp": len(dentro), "esito_calcolato": es}
        if isinstance(campo, dict) and isinstance(campo.get("metri"), dict) and str(i) in campo["metri"]:
            d["esito_campo"] = campo["metri"][str(i)].get("esito")
            d["esito"] = d["esito_campo"]; fonte = "shadow_check.json identita_per_metro"
        else:
            d["esito"] = es; fonte = fonte or "calcolato da shadow_check_scans.csv (campo identita_per_metro assente)"
        out.append(d)
    if not righe and not (isinstance(campo, dict) and campo.get("metri")):
        fonte = "non misurata: shadow_check_scans.csv assente"
    return out, fonte, S


# ---------------------------------------------------------------- R3, retries, R5
def eta_e_rifacimenti(rep, history):
    p = os.path.join(rep, "logs", "controllo_eta.jsonl")
    if not os.path.isfile(p):
        return None, "non disponibile: logs/controllo_eta.jsonl assente (controllo dell'età non acceso)"
    reg = []
    for ln in open(p):
        try:
            reg.append(json.loads(ln))
        except ValueError:
            pass
    per = {}
    for i in range(len(history)):
        R = [x for x in reg if str(x.get("finestra")) == str(i)]
        vitt = [x for x in R if x.get("ruolo") == "vittima"]
        gem = [x for x in R if x.get("ruolo") != "vittima"]
        tags = {}
        for x in gem:
            tags.setdefault(x.get("tag"), []).append(x)
        usati_guasti, ripetuti, rifatti = 0, 0, 0
        for tg, xs in tags.items():
            xs.sort(key=lambda q: q.get("tentativo") or 0)
            # failure followed by another attempt
            rifatti += sum(1 for q in xs[:-1] if q.get("esito") == "guasto")
            ult = xs[-1]
            if ult.get("esito") == "guasto":
                if (ult.get("tentativo") or 1) >= 2:
                    ripetuti += 1
                else:
                    usati_guasti += 1
        eta_v = [x.get("eta_max_s") for x in vitt if x.get("eta_max_s") is not None]
        mai_v = sum(int(x.get("n_mai_prese") or 0) for x in vitt)
        n_roll = len(tags)
        per[i] = {"rollout_gemello": n_roll, "rifatti_eta": rifatti, "guasti_ripetuti": ripetuti,
                  "guasti_usati": usati_guasti, "quota_rifacimenti": (rifatti + ripetuti) / n_roll if n_roll else None,
                  "vittima_eta_max_s": max(eta_v) if eta_v else None, "vittima_mai_prese": mai_v,
                  "vittima_voci": len(vitt),
                  "controllo_eta_history": history[i].get("controllo_eta")}
    return per, "logs/controllo_eta.jsonl"


def r5(run, rep, history):
    d = {}
    rr = os.path.join(run, "riepilogo_run.txt")
    b4 = None
    if os.path.isfile(rr):
        for ln in open(rr, errors="replace"):
            if "ESITO [4]" in ln:
                b4 = ln.strip(); break
    d["blocco4"] = (b4.split("blocco [4]:", 1)[-1].strip() if b4 else ND)
    d["blocco4_conforme"] = (("conforme" in b4) and ("non conforme" not in b4)) if b4 else None
    d["completata"] = os.path.isfile(os.path.join(rep, "completata"))
    d["finestre"] = len(history)
    ev = os.path.join(run, "eventi.txt")
    riavvii = []
    if os.path.isfile(ev):
        riavvii = [ln.strip() for ln in open(ev, errors="replace")
                   if "adattatore" in ln and re.search(r"riavvi", ln)]
    d["riavvii_adattatore"] = riavvii
    d["n_reeval_dopo_riavvio"] = sum(int(h.get("n_reevaluated_after_adapter_restart") or 0) for h in history)
    sonde = sorted(set(glob.glob(os.path.join(run, "*sonda*")) + glob.glob(os.path.join(run, "*profilo*"))
                       + glob.glob(os.path.join(rep, "logs", "*sonda*"))))
    d["file_sonda_o_profilo"] = sonde
    d["profilo_caricato"] = None   # to be read from the probe: format unknown here
    return d


# ---------------------------------------------------------------- figures
def fmt_virgola(ax):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda x, p: (f"{x:.10g}").replace(".", ","))
    ax.xaxis.set_major_formatter(f); ax.yaxis.set_major_formatter(f)


def figura_tre_passi(lati, metri, out, titolo):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, N_METRI, figsize=(5.0 * N_METRI, 13), squeeze=False)
    for k in range(N_METRI):
        if k not in lati:
            for r in range(3):
                axs[r][k].text(0.5, 0.5, ND, ha="center", transform=axs[r][k].transAxes)
            continue
        v, w = lati[k]; m = metri[k]
        for L, col, ls, nome in ((v, "tab:red", "-", "vittima"), (w, "tab:blue", "--", "vincitore")):
            tr = L["tr"]; i = L["i_stop"]
            a = axs[0][k]
            a.plot(tr["travelled"], tr["est_dist"], color=col, ls=ls, lw=2, drawstyle="steps-post", label=nome)
            if i is not None:
                a.plot(tr["travelled"][i], tr["est_dist"][i], "o", color=col, ms=8)
            b = axs[1][k]
            b.plot(tr["est_dist"], tr["cmd_v"], color=col, ls=ls, lw=2, drawstyle="steps-post", label=nome)
            c = axs[2][k]
            if i is not None:
                tt = np.asarray(tr["t"]) - tr["t"][i]
                c.plot(tt, np.hypot(tr["true_vx"], tr["true_vy"]), color=col, ls=ls, lw=2, label=f"{nome}, vera")
                c.plot(tt, tr["cmd_v"], color=col, ls=":", lw=1.2, drawstyle="steps-post", label=f"{nome}, comandata")
        a = axs[0][k]
        smx = max(v["tr"]["travelled"][-1], w["tr"]["travelled"][-1], 1.0) + 0.1
        a.plot([0, smx], [0, smx], ":", color="grey", lw=0.8); a.axhline(1.0, color="grey", lw=0.6)
        a.set_xlim(0, smx); a.set_ylim(0, max(1.3, smx))
        a.set_xlabel("strada vera percorsa [m]")
        a.set_title(f"metro w{k}\nE1 {virg(m.get('E1_allineata_cm'))} cm; R1 {virg(m['R1_cm'], segno=True)} cm\n"
                    f"R2 allineato {virg(m.get('R2_allineata_cm'))} cm, a gradino {virg(m['R2_gradino_cm'])} cm", fontsize=9)
        a.text(0.03, 0.97, f"comando di arresto a {virg(m['s_stop_vittima_m'])} m veri (vittima)\n"
                           f"e a {virg(m['s_stop_vincitore_m'])} m veri (vincitore)\n"
                           f"a robot fermo {virg(m['s_fine_vittima_m'])} e {virg(m['s_fine_vincitore_m'])} m",
               transform=a.transAxes, va="top", fontsize=8)
        b = axs[1][k]; b.set_xlabel("distanza stimata vista dal controllore [m]")
        e2 = m["E2_cm"]
        b.set_title("E2 (primo sotto crociera, ultimo non nullo, arresto):\n" +
                    " / ".join(virg(x, segno=True) for x in e2) + " cm", fontsize=9)
        c = axs[2][k]; c.axvline(0, color="k", lw=0.8); c.set_xlim(-1.5, 1.0)
        c.set_xlabel("secondi dal comando di arresto")
        c.set_title(f"velocità vera al comando: vittima {virg(m['v_vera_stop_vittima_ms'])}, vincitore {virg(m['v_vera_stop_vincitore_ms'])} m/s\n"
                    f"strada dopo il comando: {virg(m['strada_dopo_stop_vittima_cm'] and m['strada_dopo_stop_vittima_cm'] * 10, 0)} mm e "
                    f"{virg(m['strada_dopo_stop_vincitore_cm'] and m['strada_dopo_stop_vincitore_cm'] * 10, 0)} mm", fontsize=9)
        for r in range(3):
            axs[r][k].grid(alpha=0.3); fmt_virgola(axs[r][k])
    axs[0][0].set_ylabel("1) LOCALIZZAZIONE\ndistanza stimata vista dal controllore [m]")
    axs[1][0].set_ylabel("2) CONTROLLORE\ncomando di velocità [m/s]")
    axs[2][0].set_ylabel("3) SIMULAZIONE\nvelocità [m/s] (punteggiata = comando)")
    axs[0][0].legend(fontsize=8, loc="lower right"); axs[1][0].legend(fontsize=8); axs[2][0].legend(fontsize=7)
    fig.suptitle(titolo + ": vittima contro gemello vincitore nei tre passaggi localizzazione, controllore, simulazione", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out, dpi=110); plt.close(fig)


def figura_ombra(rep, R4, S, out, titolo):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pp = os.path.join(rep, "shadow_check_poses.csv")
    P = {"victim": {}, "shadow": {}}
    if os.path.isfile(pp):
        for r in csv.DictReader(open(pp)):
            try:
                P.setdefault(r["who"], {})[int(round(int(r["stamp_ns"]) / 1e8))] = (float(r["x"]), float(r["y"]))
            except (KeyError, ValueError):
                pass
    fig = plt.figure(figsize=(5.0 * N_METRI, 10))
    gs = fig.add_gridspec(2, N_METRI)
    for d in R4:
        k = d["finestra"]
        if k >= N_METRI:
            continue
        a = fig.add_subplot(gs[0, k])
        da = k10(d["da_s"]) if d["da_s"] is not None else -10 ** 9
        aa = k10(d["a_s"]) if d["a_s"] is not None else 10 ** 9
        fin = None
        for who, col, ls, nome in (("victim", "tab:red", "-", "vittima"), ("shadow", "tab:orange", "--", "ombra")):
            ks = sorted(q for q in P.get(who, {}) if da < q <= aa)
            if not ks:
                continue
            x = [P[who][q][0] for q in ks]; y = [100 * P[who][q][1] for q in ks]
            a.plot(x, y, color=col, ls=ls, lw=2, label=nome)
            a.plot(x[-1], y[-1], "s" if who == "victim" else "o", color=col, ms=9, mfc=col if who == "victim" else "none")
        comuni = sorted(q for q in P["victim"] if q in P["shadow"] and da < q <= aa)
        if comuni:
            q = comuni[-1]
            fin = 1000 * math.hypot(P["victim"][q][0] - P["shadow"][q][0], P["victim"][q][1] - P["shadow"][q][1])
        a.set_title(f"metro w{k}: {d['esito'].replace('_', ' ')}\nultima stamp comune: distano {virg(fin, 1)} mm; "
                    f"solo vittima {len(d['solo_vittima'])}, solo ombra {len(d['solo_ombra'])}", fontsize=9)
        a.set_xlabel("x stimata [m]"); a.set_ylabel("y stimata [cm]"); a.grid(alpha=0.3); fmt_virgola(a)
        if k == 0:
            a.legend(fontsize=8)
    b = fig.add_subplot(gs[1, :])
    ks = sorted(q for q in P["victim"] if q in P["shadow"])
    b.plot([q / 10 for q in ks], [1000 * math.hypot(P["victim"][q][0] - P["shadow"][q][0], P["victim"][q][1] - P["shadow"][q][1]) for q in ks],
           ".-", color="k", lw=1, ms=3, label="distanza fra la posa della vittima e quella dell'ombra (stessa stamp)")
    sv = [x for d in R4 for x in d["solo_vittima"]]; so = [x for d in R4 for x in d["solo_ombra"]]
    b.plot(so, [-0.6] * len(so), "o", mfc="none", color="tab:orange", label=f"nuvole elaborate SOLO dall'ombra ({len(so)})")
    b.plot(sv, [-1.2] * len(sv), "s", color="tab:red", label=f"nuvole elaborate SOLO dalla vittima ({len(sv)})")
    for d in R4:
        lo = d["da_s"] if d["da_s"] is not None else (ks[0] / 10 if ks else 0)
        hi = d["a_s"] if d["a_s"] is not None else (ks[-1] / 10 if ks else 0)
        b.axvspan(lo, hi, color="grey", alpha=0.08 if d["finestra"] % 2 == 0 else 0.18)
        b.text((lo + hi) / 2, b.get_ylim()[1] if False else 0.97, f"w{d['finestra']}", transform=b.get_xaxis_transform(), ha="center", va="top")
    n_div = sum(1 for d in R4 if d["esito"] == "ombra_divergente")
    b.set_title(f"R4: ombra divergente in {n_div} metri su {len(R4)} (metro k = stamp della vittima in (open_stamp_k, open_stamp_k+1])", fontsize=10)
    b.set_xlabel("tempo simulato (stamp della nuvola) [s]"); b.set_ylabel("distanza vittima-ombra [mm]")
    b.grid(alpha=0.3); b.legend(fontsize=8); fmt_virgola(b)
    fig.suptitle(titolo + ": ombra contro vittima, stime di MOLA nel frame della mappa", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out, dpi=110); plt.close(fig)


# ---------------------------------------------------------------- main
def trova_rep(run):
    ej = os.path.join(run, "esito_run.json")
    if os.path.isfile(ej):
        try:
            r = json.load(open(ej)).get("fatti", {}).get("rep")
            if r and os.path.isdir(r):
                return r
        except (OSError, ValueError):
            pass
    c = sorted(glob.glob(os.path.join(run, "campagna", "*", "*", "rep_*")))
    if os.path.isfile(os.path.join(run, "history.json")):
        return run
    return c[-1] if c else None


def giudizio(val, soglia, assoluto=True):
    if val is None:
        return ND
    return PASSA if (abs(val) if assoluto else val) <= soglia else NON_PASSA


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="cartella della run (seme), con frame_log.csv e campagna/.../rep_NN")
    ap.add_argument("--out", required=True, help="cartella di uscita nuova (non deve esistere o essere vuota)")
    ap.add_argument("--titolo", default=None)
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    if os.path.isdir(a.out) and os.listdir(a.out):
        sys.exit(f"--out {a.out} esiste e non e' vuota: scegliere una cartella nuova")
    os.makedirs(a.out, exist_ok=True)
    rep = trova_rep(run)
    if rep is None or not os.path.isfile(os.path.join(rep, "history.json")):
        sys.exit("history.json non trovato")
    history = json.load(open(os.path.join(rep, "history.json")))
    titolo = a.titolo or os.path.relpath(run, os.path.expanduser("~/mola-adversarial-nsga3"))
    fl_path = os.path.join(run, "frame_log.csv")
    fl = leggi_frame_log(fl_path) if os.path.isfile(fl_path) else None
    log_v = leggi_log_mola(os.path.join(rep, "logs", "mola_victim.log"))
    lati, metri, fonti = {}, {}, {}
    for h in history:
        k = int(h["window"])
        idx = h.get("vincitore_idx")
        pv = os.path.join(rep, "traces", f"w{k}_applied.csv")
        pw = os.path.join(rep, "traces", f"w{k}_e{int(idx):02d}.csv") if idx is not None else None
        lw = os.path.join(rep, "logs", f"mola_w{k}_e{int(idx)}.log") if idx is not None else None
        if not (pw and os.path.isfile(pv) and os.path.isfile(pw) and fl is not None and os.path.isfile(lw)):
            metri[k] = {"errore": "tracce, log o frame_log mancanti"}; continue
        v = prepara_lato(leggi_traccia(pv), log_v, fl, True)
        logw = leggi_log_mola(lw)
        w = prepara_lato(leggi_traccia(pw), logw, fl, False)
        lati[k] = (v, w)
        m = misura_metro(v, w)
        m["vincitore_idx"] = int(idx)
        m["warn_tf_ricuciti_vincitore"] = logw["warn_tf"]
        # R1 on the re-evaluation needs a trace with travelled from the re-evaluation
        rv = sorted(glob.glob(os.path.join(rep, "traces", f"w{k}_sel*.csv")) + glob.glob(os.path.join(rep, "traces", f"w{k}_r*.csv")))
        rv = [p for p in rv if not p.endswith("_applied.csv")]
        m["R1_rivalutazione_cm"] = None
        m["R1_rivalutazione_nota"] = ND + ": nessuna traccia con travelled della rivalutazione in traces/ (logs/traces_wK_selJ.csv sono tracce interne di MOLA)"
        if rv:
            vals = []
            for p in rv:
                t = leggi_traccia(p)
                if "travelled" in t:
                    vals.append((os.path.basename(p), 100 * (v["tr"]["travelled"][-1] - t["travelled"][-1])))
            if vals:
                m["R1_rivalutazione_cm"] = vals; m["R1_rivalutazione_nota"] = "tracce " + ", ".join(n for n, _ in vals)
        fonti[k] = {"vittima": os.path.relpath(pv, run), "vincitore": os.path.relpath(pw, run),
                    "log_vincitore": os.path.relpath(lw, run)}
        metri[k] = m
    R4, fonte_r4, S = ombra_per_metro(rep, history)
    eta, fonte_eta = eta_e_rifacimenti(rep, history)
    v5 = r5(run, rep, history)

    # ---- per-metre verdicts
    tab = []
    for k in range(N_METRI):
        m = metri.get(k, {})
        if "R1_cm" not in m:
            tab.append({"metro": k, "errore": m.get("errore", "metro assente")}); continue
        g = {"metro": k}
        g["R1"] = {"valore_cm": m["R1_cm"], "soglia_cm": SOGLIA_R1_CM, "esito": giudizio(m["R1_cm"], SOGLIA_R1_CM),
                   "esito_4cm_info": giudizio(m["R1_cm"], SOGLIA_R1_INFO_CM),
                   "rivalutazione": m["R1_rivalutazione_cm"], "nota_rivalutazione": m["R1_rivalutazione_nota"]}
        g["E1"] = {"valore_cm": m.get("E1_allineata_cm"), "soglia_cm": SOGLIA_E1_CM,
                   "esito": giudizio(m.get("E1_allineata_cm"), SOGLIA_E1_CM),
                   "stessa_stamp_cm": m.get("E1_stessa_stamp_cm")}
        d4 = R4[k] if k < len(R4) else None
        if d4 is None or d4["esito"] == "non_misurata":
            e4 = ND
        else:
            e4 = PASSA if d4["esito"] == "identica" else NON_PASSA
        g["R4"] = {"esito_ombra": d4["esito"] if d4 else "non_misurata", "esito": e4,
                   "solo_vittima": d4["solo_vittima"] if d4 else None, "solo_ombra": d4["solo_ombra"] if d4 else None,
                   "max_dxy_m": d4["max_dxy_m"] if d4 else None, "soglia_m": SOGLIA_R4_XY_M}
        if eta is None:
            g["R3"] = {"esito": ND, "nota": fonte_eta}
        else:
            e = eta.get(k, {})
            ok = (e.get("guasti_usati") == 0 and e.get("vittima_voci", 0) > 0 and e.get("vittima_mai_prese") == 0
                  and e.get("vittima_eta_max_s") is not None and e["vittima_eta_max_s"] <= SOGLIA_ETA_S)
            g["R3"] = {**e, "soglia_s": SOGLIA_ETA_S,
                       "esito": (ND if not e.get("vittima_voci") else (PASSA if ok else NON_PASSA))}
            q = e.get("quota_rifacimenti")
            g["limite_rifacimenti"] = {"quota": q, "soglia": SOGLIA_RIFACIMENTI,
                                       "superato": (q is not None and q > SOGLIA_RIFACIMENTI)}
        g["R2_allineato"] = {"valore_cm": m.get("R2_allineata_cm"), "soglia_cm": SOGLIA_R2_ALL_CM,
                             "esito": giudizio(m.get("R2_allineata_cm"), SOGLIA_R2_ALL_CM), "riportato": True}
        g["R2_gradino"] = {"valore_cm": m["R2_gradino_cm"], "soglia_cm": SOGLIA_R2_GRAD_CM,
                           "esito": giudizio(m["R2_gradino_cm"], SOGLIA_R2_GRAD_CM), "riportato": True}
        g["R2_interpolato"] = {"valore_cm": m.get("R2_interpolato_cm"), "soglia_cm": SOGLIA_R2_GRAD_CM,
                               "esito": giudizio(m.get("R2_interpolato_cm"), SOGLIA_R2_GRAD_CM), "riportato": True}
        g["R2_stessa_stamp"] = {"valore_cm": m.get("R2_stessa_stamp_cm"), "riportato": True}
        e2 = m["E2_cm"]
        g["E2"] = {"valori_cm": e2, "soglia_cm": SOGLIA_E2_CM,
                   "esito": (ND if any(x is None for x in e2) else (PASSA if all(abs(x) <= SOGLIA_E2_CM for x in e2) else NON_PASSA))}
        ds = None if m["strada_dopo_stop_vittima_cm"] is None or m["strada_dopo_stop_vincitore_cm"] is None else \
            m["strada_dopo_stop_vittima_cm"] - m["strada_dopo_stop_vincitore_cm"]
        dv = None if m["v_vera_stop_vittima_ms"] is None or m["v_vera_stop_vincitore_ms"] is None else \
            m["v_vera_stop_vittima_ms"] - m["v_vera_stop_vincitore_ms"]
        g["E3"] = {"delta_strada_dopo_stop_cm": ds, "soglia_cm": SOGLIA_E3_STRADA_CM,
                   "delta_v_vera_stop_ms": dv, "soglia_ms": SOGLIA_E3_V_MS,
                   "esito": (ND if ds is None or dv is None else
                             (PASSA if abs(ds) <= SOGLIA_E3_STRADA_CM and abs(dv) <= SOGLIA_E3_V_MS else NON_PASSA)),
                   "ruolo": "controllo di sanità"}
        tab.append(g)

    # ---- overall outcome
    motivi_nv, motivi_np, mancanti = [], [], []
    if not v5["completata"] or v5["finestre"] != N_METRI:
        motivi_nv.append(f"run non completata ({v5['finestre']} finestre, file completata {'presente' if v5['completata'] else 'assente'})")
    if fonte_r4.startswith("non misurata") or any(g.get("R4", {}).get("esito_ombra") == "non_misurata" for g in tab):
        motivi_nv.append("ombra non misurata in almeno un metro: run non valida (emendamento 3)")
    for g in tab:
        if "errore" in g:
            motivi_nv.append(f"w{g['metro']}: {g['errore']}"); continue
        if g.get("limite_rifacimenti", {}).get("superato"):
            motivi_nv.append(f"w{g['metro']}: rifacimenti per età {virg(100 * g['limite_rifacimenti']['quota'], 1)} % > 5 %")
        for c in ("R1", "E1", "R4", "R3"):
            e = g[c]["esito"]
            if e == NON_PASSA:
                det = {"R1": lambda: f"{virg(g['R1']['valore_cm'], segno=True)} cm",
                       "E1": lambda: f"{virg(g['E1']['valore_cm'])} cm",
                       "R4": lambda: f"{g['R4']['esito_ombra']}, solo vittima {len(g['R4']['solo_vittima'] or [])}, solo ombra {len(g['R4']['solo_ombra'] or [])}, max dxy {virg((g['R4']['max_dxy_m'] or 0) * 1000, 2)} mm",
                       "R3": lambda: "età o nuvole mai prese"}[c]()
                motivi_np.append(f"{c} w{g['metro']}: {det}")
            elif e == ND:
                mancanti.append(f"{c} w{g['metro']}")
    if v5["blocco4_conforme"] is False:
        motivi_np.append("R5: blocco [4] non conforme")
    elif v5["blocco4_conforme"] is None:
        mancanti.append("R5 blocco [4]")
    if v5["riavvii_adattatore"]:
        motivi_np.append(f"R5: {len(v5['riavvii_adattatore'])} riavvii dell'adattatore in eventi.txt (verificare se a Isaac in play)")
    if v5["profilo_caricato"] is None:
        mancanti.append("R5 profilo DDS caricato da tutti i processi")
    if motivi_nv:
        esito = "NON VALUTABILE"
    elif motivi_np:
        esito = "NON PASSA"
    elif mancanti:
        esito = "NON VALUTABILE"
    else:
        esito = "PASSA"
    divergenti = [g["metro"] for g in tab if g.get("R4", {}).get("esito_ombra") == "ombra_divergente"]
    note = []
    if divergenti:
        note.append(f"ombra divergente in {len(divergenti)}/{N_METRI} metri: per l'emendamento 3 è un guasto catena con rifacimento dell'INTERA run")
    note.append("E2 ed E3 riportati con le soglie confermate; non entrano nell'esito (emendamento 3: principali R1, E1, R4, R3; validità R5 e limite dei rifacimenti)")
    note.append("R2 riportato (allineato al moto, 5 cm), con gradino e interpolato (3 cm): non entra nell'esito")

    res = {"run": run, "rep": rep, "esito_complessivo": esito, "motivi_non_valutabile": motivi_nv,
           "motivi_non_passa": motivi_np, "criteri_non_disponibili": mancanti, "note": note,
           "fonti": {"metri": fonti, "R4": fonte_r4, "R3": fonte_eta, "frame_log": os.path.relpath(fl_path, run),
                     "log_vittima": os.path.relpath(os.path.join(rep, "logs", "mola_victim.log"), run)},
           "criteri_per_metro": tab, "misure_per_metro": {str(k): {kk: vv for kk, vv in m.items() if not kk.startswith("serie_")}
                                                          for k, m in metri.items()},
           "R4_per_metro": R4, "R5": v5, "eta_per_metro": eta,
           "log_vittima": {"warn_tf_ricuciti": log_v["warn_tf"], "pathStep_isGood_senza_posa": log_v["icp_senza_posa"]}}
    json.dump(res, open(os.path.join(a.out, "esito.json"), "w"), indent=1, default=float)

    # ---- esito.md
    L = [f"# Prova di accettazione: esito di {titolo}", "",
         f"Script `analisi_accettazione.py`; criterio `criterio_prova_accettazione.md` con gli emendamenti 1-3 (il 3 prevale).",
         f"Run `{run}`; rep `{os.path.relpath(rep, run)}`; n = {len(history)} metri (history.json).", "",
         f"## Esito complessivo: **{esito}**", ""]
    for t, xs in (("Non valutabile perché", motivi_nv), ("Non passa per", motivi_np), ("Criteri non disponibili", mancanti), ("Note", note)):
        if xs:
            L.append(f"{t}:"); L += [f"- {x}" for x in xs]; L.append("")
    L += ["## Criteri principali per metro (vittima − vincitore)", "",
          "| metro | R1 [cm] (≤ 3,0) | R1 con 4 cm (info) | E1 [cm] (≤ 3,0) | R4 ombra | R3 età |",
          "|---|---|---|---|---|---|"]
    for g in tab:
        if "errore" in g:
            L.append(f"| w{g['metro']} | {g['errore']} | | | | |"); continue
        r4 = g["R4"]
        r4t = f"{r4['esito']} ({r4['esito_ombra']}; solo vittima {len(r4['solo_vittima'] or [])}, solo ombra {len(r4['solo_ombra'] or [])}, max dxy {virg((r4['max_dxy_m'] or 0) * 1000, 2)} mm)"
        L.append(f"| w{g['metro']} | {virg(g['R1']['valore_cm'], segno=True)} {g['R1']['esito']} | {g['R1']['esito_4cm_info']} | "
                 f"{virg(g['E1']['valore_cm'])} {g['E1']['esito']} | {r4t} | {g['R3']['esito']} |")
    L += ["", "R1 sulla rivalutazione: " + (tab[0]["R1"]["nota_rivalutazione"] if tab and "R1" in tab[0] else ND), "",
          "## Riportati", "",
          "| metro | R2 allineato [cm] (5,0) | R2 a gradino [cm] (3,0) | R2 interpolato [cm] (3,0) | R2 stessa stamp [cm] | E2 tre punti [cm] (3,0) | E3 Δ strada dopo arresto [cm] (1,0) / Δ v vera [m/s] (0,05) |",
          "|---|---|---|---|---|---|---|"]
    for g in tab:
        if "errore" in g:
            continue
        L.append(f"| w{g['metro']} | {virg(g['R2_allineato']['valore_cm'])} {g['R2_allineato']['esito']} | "
                 f"{virg(g['R2_gradino']['valore_cm'])} {g['R2_gradino']['esito']} | {virg(g['R2_interpolato']['valore_cm'])} {g['R2_interpolato']['esito']} | "
                 f"{virg(g['R2_stessa_stamp']['valore_cm'])} | {' / '.join(virg(x, segno=True) for x in g['E2']['valori_cm'])} {g['E2']['esito']} | "
                 f"{virg(g['E3']['delta_strada_dopo_stop_cm'], segno=True)} / {virg(g['E3']['delta_v_vera_stop_ms'], segno=True)} {g['E3']['esito']} |")
    L += ["", "## Grandezze di base per metro", "",
          "| metro | vincitore_idx | strada al comando di arresto vittima / vincitore [m] | a robot fermo [m] | v vera al comando [m/s] | strada dopo il comando [mm] | spostamento orologio [s] | stamp comuni allineate | associazione per valore mediana / max [cm] (vitt; vinc) | nuvole non elaborate vitt / vinc |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k in range(N_METRI):
        m = metri.get(k, {})
        if "R1_cm" not in m:
            continue
        L.append(f"| w{k} | {m['vincitore_idx']} | {virg(m['s_stop_vittima_m'])} / {virg(m['s_stop_vincitore_m'])} | "
                 f"{virg(m['s_fine_vittima_m'], 3)} / {virg(m['s_fine_vincitore_m'], 3)} | {virg(m['v_vera_stop_vittima_ms'])} / {virg(m['v_vera_stop_vincitore_ms'])} | "
                 f"{virg(m['strada_dopo_stop_vittima_cm'] * 10, 0)} / {virg(m['strada_dopo_stop_vincitore_cm'] * 10, 0)} | "
                 f"{virg(m.get('spostamento_orologio_allineata_s'), 1)} | {m.get('n_comuni_allineata')} | "
                 f"{virg(m['assoc_valore_med_cm_vittima'])} / {virg(m['assoc_valore_max_cm_vittima'])}; {virg(m['assoc_valore_med_cm_vincitore'])} / {virg(m['assoc_valore_max_cm_vincitore'])} | "
                 f"{m['non_elaborate_vittima']}/{m['pubblicate_vittima']} / {m['non_elaborate_vincitore']}/{m['pubblicate_vincitore']} |")
    L += ["", "## R5 e limite dei rifacimenti", "",
          f"- blocco [4]: {v5['blocco4']} (riepilogo_run.txt)",
          f"- run completata: {'sì' if v5['completata'] else 'no'} (file completata); finestre in history.json: {v5['finestre']}",
          f"- riavvii dell'adattatore in eventi.txt: {len(v5['riavvii_adattatore'])}; rivalutazioni dopo riavvio (history): {v5['n_reeval_dopo_riavvio']}",
          f"- profilo DDS caricato da tutti i processi: {ND} (file della sonda o del profilo trovati: {', '.join(v5['file_sonda_o_profilo']) or 'nessuno'})",
          f"- R3 e rifacimenti: fonte {fonte_eta}", "",
          "## Definizioni usate (dichiarate prima dei dati della prova)", "",
          "- Vittima `traces/wK_applied.csv`, vincitore `traces/wK_e{vincitore_idx:02d}.csv` (history.json `vincitore_idx`, traccia di ricerca); strada vera `travelled`, stima `est_dist`, comando `cmd_v`, velocità vera √(true_vx² + true_vy²).",
          "- Comando di arresto: prima riga con `cmd_v` = 0 dopo la prima riga con `cmd_v` > 0. s_fine = `travelled` dell'ultima riga (robot fermo).",
          "- Posa di MOLA per stamp: ultima riga `ICP (kind=0)` o `ICP (kind=1)` prima della riga `pathStep` con `isGood=1` di quello stamp, nel log ricucito (tolti i WARN TF_OLD_DATA con la riga 'Possible reasons...'). Vittima `logs/mola_victim.log`, vincitore `logs/mola_wK_e{idx}.log`.",
          "- Nuvole del metro: righe di `frame_log.csv` con `t_pub_end_wall` fra W0 = wall − t della prima riga e il wall dell'ultima; vittima `victim_stream` = 1 (`stamp_victim`), vincitore `victim_stream` = 0 (`stamp_main_out`). Posa vera alla stamp: `true_x`, `true_y` interpolati al `t_dds_src` della nuvola.",
          "- Errore alla stessa stamp: |(p_v − p_w) − R(ψ)(T_v − T_w)|, ψ dalla direzione dello spostamento della vittima nel metro (MOLA contro vero). Allineato al moto: stamp del vincitore spostato di stamp_W0_vinc − stamp_W0_vitt (stamp dell'ultima nuvola pubblicata prima di W0).",
          "- Posa in mano al controllore: associazione per valore fra `est_dist` e |p(stamp) − p(stamp di riferimento)|, stamp non decrescente e nuvola partita prima della riga. E1: ultima stamp comune allineata ≤ min(stamp in mano ai due al comando di arresto).",
          "- R2 a gradino: griglia di 1 cm su `travelled` fino a min(s_fine), d(s) = ultimo `est_dist` a `travelled` ≤ s. R2 interpolato: posa di MOLA collocata sulla strada vera al `t_dds_src` della sua nuvola e interpolata linearmente sulla stessa griglia.",
          "- E2: `est_dist` al primo `cmd_v` sotto la crociera (cmd_v della prima riga), all'ultimo `cmd_v` ≠ 0, al comando di arresto. E3: strada dopo il comando e velocità vera al comando.",
          "- R4: campo `identita_per_metro` di shadow_check.json se c'è; altrimenti da shadow_check_scans.csv con la regola del codice nuovo (stamp elaborata da una sola = divergente; |dxy| > 1e-6 m = divergente; metro k = stamp in (open_stamp_k, open_stamp_k+1] di history.json; stamp oltre l'ultima elaborata da tutte e due esclusi).",
          "- R3: `logs/controllo_eta.jsonl` (registro per rollout del codice nuovo): vittima senza età > 0,40 s né nuvole mai prese; nessun rollout del gemello con l'ultimo tentativo in guasto e senza rifacimento. Limite: (rifatti + ripetuti) / rollout distinti del gemello nel metro ≤ 5 %.", ""]
    open(os.path.join(a.out, "esito.md"), "w").write("\n".join(L))

    figura_tre_passi(lati, metri, os.path.join(a.out, "tre_passaggi.png"), titolo)
    figura_ombra(rep, R4, S, os.path.join(a.out, "ombra_contro_vittima.png"), titolo)
    # CSV of the same-stamp series, next to the figures
    with open(os.path.join(a.out, "serie_stessa_stamp.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["metro", "variante", "stamp_vittima_s", "errore_diff_cm"])
        for k, m in metri.items():
            for var in ("stessa_stamp", "allineata"):
                for s, e in m.get(f"serie_{var}", []):
                    wr.writerow([k, var, f"{s:.1f}", f"{e:.4f}"])
    print(f"esito complessivo: {esito}")
    for x in motivi_nv + motivi_np:
        print("  -", x)
    print(f"uscite in {a.out}")


if __name__ == "__main__":
    main()
