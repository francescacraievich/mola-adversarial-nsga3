"""Latency instrumentation inside perturbation_node (--strumenti-ritardo CSV).

Off by default: the node imports this module only with the option, so without it
there are no extra imports, threads, gc callbacks or faulthandler timers.

Files (same prefix as the given CSV; if a file exists, e.g. node restarted on the
same output, _1, _2, ... is appended as for the frame log):
  <p>.csv        one row per received scan (COLONNE): DDS times from MessageInfo,
                 callback start/end, end of each phase, take/release in the
                 publisher thread, start, publish() call and end per output topic;
  <p>_gc.csv     one row per garbage collection (COLONNE_GC);
  <p>_cpu.csv    /proc/stat, /proc/loadavg, /proc/pressure/cpu and /proc/self/stat
                 samples every PERIODO_CPU_S (COLONNE_CPU);
  <p>_cane.csv   watchdog events (COLONNE_CANE);
  <p>_stack.txt  stacks of all Python threads written by faulthandler, each
                 followed by a "--- stack N ..." line with the time and the thread
                 table (name, hex ident, native id).

Times: *_mono_ns time.monotonic_ns(), *_wall_ns time.time_ns(), taken together.
dds_recv_ns and dds_src_ns are MessageInfo received_timestamp and source_timestamp
(rmw clock, ns, unconverted).

Watchdog: faulthandler.dump_traceback_later runs a C thread that dumps stacks
without taking the GIL, so it also catches a thread holding the GIL, a gc pass or a
stuck C extension. There is one timer per process, armed only by the scan callback,
which runs one at a time (mutually exclusive group):
  - "durata": armed at callback start, fires if it lasts more than SOGLIA_S;
  - "attesa": armed at callback end, fires if the next callback does not start
    within SOGLIA_S (if only the input was missing, the next scan row shows
    ritardo_ricezione_ms below threshold).
At most one stack every MIN_INTERVALLO_STACK_S; MAX_STACK real stacks and
MAX_STACK_TOTALE overall. A false wait (input gap) counts neither towards the real
cap nor the rate limit (attesa_falsa row in _cane.csv). Over-threshold events
without a stack (limits hit) are still logged in _cane.csv.

Writing: all rows go to in-memory queues; a single thread flushes them every
FLUSH_S. The only write outside that thread is the "--- stack" line after a fired
stack (written by the callback once the dump is complete)."""
import faulthandler
import gc
import os
import threading
import time
from collections import deque
from pathlib import Path

SOGLIA_S = 0.200
MIN_INTERVALLO_STACK_S = 0.5
MAX_STACK = 500              # real stacks (false waits excluded)
MAX_STACK_TOTALE = 20000     # all stacks, false waits included: disk cap
MARGINE_VERIFICA_NS = 5_000_000
PERIODO_CPU_S = 0.25
FLUSH_S = 1.0

TOPIC = ("main", "victim", "attacker", "shadow")
EVENTI = (("cb_inizio", "filtro_fine", "lettura_fine", "perturbazione_fine", "ramo_fine",
           "coerenza_fine", "accodata", "cb_fine", "presa", "rilascio")
          + tuple(f"{t}_{e}" for t in TOPIC for e in ("inizio", "chiamata", "fine"))
          + ("registri_fine",))
COLONNE = (("seq", "frame_idx", "stamp_in", "n_punti", "esito", "tid_cb", "tid_pub",
            "dds_src_ns", "dds_recv_ns", "pub_seq", "ric_seq",
            "ritardo_ricezione_ms", "ritardo_dds_ms", "durata_cb_ms", "stack_attesa", "stack_durata")
           + tuple(f"{e}_{c}" for e in EVENTI for c in ("mono_ns", "wall_ns")))
COLONNE_GC = ("inizio_mono_ns", "inizio_wall_ns", "fine_mono_ns", "fine_wall_ns", "durata_ms",
              "generazione", "raccolti", "non_raccoglibili", "tid")
COLONNE_CPU = ("mono_ns", "wall_ns", "cpu_user", "cpu_nice", "cpu_system", "cpu_idle", "cpu_iowait",
               "cpu_irq", "cpu_softirq", "cpu_steal", "procs_running", "procs_blocked",
               "load1", "load5", "load15", "runnable_su_totale", "psi_cpu_some_total_us",
               "psi_cpu_full_total_us", "nodo_utime_tick", "nodo_stime_tick", "nodo_thread")
COLONNE_CANE = ("evento", "seq", "tipo", "mono_ns", "wall_ns", "valore_ms", "n_stack",
                "byte_inizio", "byte_fine")


def _csv(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return repr(v) if v == v else ""
    return str(v)


def percorsi_liberi(path):
    """Prefix not yet used by any of the five files."""
    path = Path(path)
    stem, suf, cart = path.stem, path.suffix or ".csv", path.parent

    def files(s):
        return {"righe": cart / f"{s}{suf}", "gc": cart / f"{s}_gc.csv", "cpu": cart / f"{s}_cpu.csv",
                "cane": cart / f"{s}_cane.csv", "stack": cart / f"{s}_stack.txt"}
    f, i = files(stem), 0
    while any(p.exists() for p in f.values()):
        i += 1
        f = files(f"{stem}_{i}")
    return f


class _Scrittore:
    """One thread for all CSVs: rows queued (deque, no I/O in the caller), written
    every FLUSH_S; chiudi() writes the rest."""

    def __init__(self, percorsi_colonne):
        self._file = {}
        for nome, (p, col) in percorsi_colonne.items():
            f = open(p, "w", buffering=1 << 16)
            f.write(",".join(col) + "\n")
            f.flush()
            self._file[nome] = (f, deque(), len(col))
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, name="strumenti_scrittura", daemon=True)
        self._t.start()

    def add(self, nome, riga):
        self._file[nome][1].append(riga)

    def _drain(self):
        for f, q, n in self._file.values():
            righe = []
            while True:
                try:
                    r = q.popleft()
                except IndexError:
                    break
                r = list(r) + [None] * (n - len(r))
                righe.append(",".join(_csv(v) for v in r))
            if righe:
                f.write("\n".join(righe) + "\n")
                f.flush()

    def _loop(self):
        while not self._stop.wait(FLUSH_S):
            self._drain()

    def chiudi(self):
        self._stop.set()
        self._t.join(timeout=5.0)
        self._drain()
        for f, _, _ in self._file.values():
            f.close()


def _adesso():
    return time.monotonic_ns(), time.time_ns()


class Nuvola:
    """Timestamps of one scan, from the callback to the end of publishing."""
    __slots__ = ("seq", "t", "dati", "parti", "stack_attesa", "stack_durata")

    def __init__(self, seq):
        self.seq = seq
        self.t = {}
        self.dati = {}
        self.parti = 1          # 2 if queued: callback and publisher thread
        self.stack_attesa = None
        self.stack_durata = None

    def fase(self, nome):
        self.t[nome] = _adesso()


class _Cane:
    """faulthandler timer armed and checked only by the scan callback.

    Caps: MAX_STACK real stacks and MAX_STACK_TOTALE overall. A false wait (input
    gap: the next scan has ritardo_ricezione_ms below threshold) stays in the file
    but uses neither the real cap nor the rate limit."""

    def __init__(self, percorso_stack, scrittore, soglia_s=SOGLIA_S,
                 min_intervallo_s=MIN_INTERVALLO_STACK_S, max_stack=MAX_STACK,
                 max_totale=MAX_STACK_TOTALE):
        self._f = open(percorso_stack, "ab", buffering=0)
        self._f.write(f"# stack del cane da guardia (soglia {soglia_s * 1000:.0f} ms), pid {os.getpid()}\n"
                      .encode())
        self._byte = os.fstat(self._f.fileno()).st_size   # known size: no fstat when arming
        self._scr = scrittore
        self.soglia_s = soglia_s
        self._soglia_ns = int(round(soglia_s * 1e9))
        self._min_ns = int(round(min_intervallo_s * 1e9))
        self._max = max_stack
        self._max_tot = max_totale
        self._lock = threading.Lock()
        self._armato = None      # (type, seq, mono_ns, wall_ns, size before)
        self._ultimo_ns = None
        self._ultimo_prec = None
        self.n_stack = 0         # stacks written (numbering in the file)
        self.n_veri = 0          # stacks counted against MAX_STACK
        self.n_falsi = 0
        self.n_saltati = 0
        self._chiuso = False

    def _verifica(self):
        """Disarm; if the stack fired, write the line with its time and return
        (type, number). Call with the lock held."""
        if self._armato is None:
            return None
        faulthandler.cancel_dump_traceback_later()   # waits for a dump in progress
        tipo, seq, m0, w0, b0 = self._armato
        self._armato = None
        # before the threshold the timer cannot have fired: skip the syscall
        if time.monotonic_ns() - m0 < self._soglia_ns - MARGINE_VERIFICA_NS:
            return None
        b1 = os.fstat(self._f.fileno()).st_size
        if b1 <= b0:
            return None
        self.n_stack += 1
        self.n_veri += 1
        m, w = m0 + self._soglia_ns, w0 + self._soglia_ns
        self._ultimo_prec, self._ultimo_ns = self._ultimo_ns, m
        # a just-started thread (still in _limbo) has ident None: written as "-"
        thr = " ".join(f"{t.name}:{t.ident:#x}:{t.native_id}" if t.ident is not None
                       else f"{t.name}:-:-" for t in threading.enumerate())
        self._f.write((f"--- stack {self.n_stack} tipo={tipo} nuvola={seq} scatto_mono_ns={m} "
                       f"scatto_wall_ns={w} armato_mono_ns={m0} byte={b0}-{b1}\n"
                       f"--- thread {thr}\n").encode())
        self._byte = os.fstat(self._f.fileno()).st_size
        self._scr.add("cane", ("stack", seq, tipo, m, w, None, self.n_stack, b0, b1))
        return tipo, self.n_stack

    def arma(self, tipo, seq, attesa_falsa=False):
        """Disarm the previous timer (returning the fired stack, if any) and arm a
        new one unless the rate limit or caps apply. attesa_falsa: the arming scan
        reached the node on time, so an "attesa" stack that just fired was an input
        gap."""
        with self._lock:
            if self._chiuso:          # callback finished after node shutdown
                return None
            scattato = self._verifica()
            if scattato is not None and scattato[0] == "attesa" and attesa_falsa:
                self.n_veri -= 1
                self.n_falsi += 1
                self._ultimo_ns = self._ultimo_prec
                self._scr.add("cane", ("attesa_falsa", seq, "attesa", None, None, None, scattato[1]))
            m, w = _adesso()
            if (self.n_veri >= self._max or self.n_stack >= self._max_tot
                    or (self._ultimo_ns is not None and m - self._ultimo_ns < self._min_ns)):
                self.n_saltati += 1
                return scattato
            self._armato = (tipo, seq, m, w, self._byte)
            faulthandler.dump_traceback_later(self.soglia_s, repeat=False, file=self._f, exit=False)
            return scattato

    def disarma(self):
        with self._lock:
            return None if self._chiuso else self._verifica()

    def chiudi(self):
        with self._lock:
            if self._chiuso:
                return
            self._verifica()
            self._chiuso = True
            self._f.close()


class StrumentiRitardo:
    """Per-scan rows, garbage collector, CPU sampler and watchdog."""

    def __init__(self, path, soglia_s=SOGLIA_S, periodo_cpu_s=PERIODO_CPU_S):
        p = percorsi_liberi(path)
        p["righe"].parent.mkdir(parents=True, exist_ok=True)
        self.percorsi = p
        self.path = p["righe"]
        self.soglia_s = soglia_s
        self._soglia_ns = int(round(soglia_s * 1e9))
        self._scr = _Scrittore({"righe": (p["righe"], COLONNE), "gc": (p["gc"], COLONNE_GC),
                                "cpu": (p["cpu"], COLONNE_CPU), "cane": (p["cane"], COLONNE_CANE)})
        self.cane = _Cane(p["stack"], self._scr, soglia_s=soglia_s)
        self._lock = threading.Lock()
        self._seq = 0
        self._locale = threading.local()
        self._gc_t0 = None
        gc.callbacks.append(self._gc_cb)
        self._stop = threading.Event()
        self._periodo_cpu = periodo_cpu_s
        self._t_cpu = threading.Thread(target=self._cpu_loop, name="strumenti_cpu", daemon=True)
        self._t_cpu.start()
        self._chiuso = False

    # --- scan callback (one at a time) -----------------------------------------

    def inizio_callback(self, info, msg, frame_idx):
        r = Nuvola(self._seq)
        self._seq += 1
        r.fase("cb_inizio")
        r.dati["tid_cb"] = threading.get_native_id()
        r.dati["frame_idx"] = frame_idx
        try:
            r.dati["stamp_in"] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            r.dati["n_punti"] = msg.width * msg.height
        except Exception:
            pass
        if isinstance(info, dict):
            r.dati["dds_src_ns"] = info.get("source_timestamp")
            r.dati["dds_recv_ns"] = info.get("received_timestamp")
            r.dati["pub_seq"] = info.get("publication_sequence_number")
            r.dati["ric_seq"] = info.get("reception_sequence_number")
        recv, src = r.dati.get("dds_recv_ns"), r.dati.get("dds_src_ns")
        rit = (r.t["cb_inizio"][1] - recv) / 1e6 if recv else None
        if recv and src:
            # transport plus wait inside the node's Fast DDS (until history insertion)
            r.dati["ritardo_dds_ms"] = (recv - src) / 1e6
        scattato = self.cane.arma("durata", r.seq,
                                  attesa_falsa=rit is not None and rit < self.soglia_s * 1000.0)
        if scattato is not None and scattato[0] == "attesa":
            r.stack_attesa = scattato[1]
        if rit is not None:
            r.dati["ritardo_ricezione_ms"] = rit
            if rit > self.soglia_s * 1000.0:
                self._scr.add("cane", ("ritardo_inizio", r.seq, None, r.t["cb_inizio"][0],
                                       r.t["cb_inizio"][1], rit, r.stack_attesa))
        return r

    def fine_callback(self, r, accodata):
        r.fase("cb_fine")
        scattato = self.cane.arma("attesa", r.seq)
        if scattato is not None:
            r.stack_durata = scattato[1]
        dur = (r.t["cb_fine"][0] - r.t["cb_inizio"][0]) / 1e6
        r.dati["durata_cb_ms"] = dur
        if dur > self.soglia_s * 1000.0:
            self._scr.add("cane", ("durata", r.seq, None, r.t["cb_inizio"][0], r.t["cb_inizio"][1],
                                   dur, r.stack_durata))
        if not accodata:
            r.dati.setdefault("esito", "senza_uscita")
        self._fine_parte(r)

    # --- publisher thread -------------------------------------------------------

    def presa(self, r):
        r.fase("presa")
        r.dati["tid_pub"] = threading.get_native_id()

    def pub_inizio(self, r, topic):
        self._locale.r, self._locale.topic = r, topic
        r.fase(f"{topic}_inizio")

    def chiamata(self):
        """Right before publish() (message already built). No-op without a current
        scan in this thread (replay, other threads)."""
        r = getattr(self._locale, "r", None)
        if r is not None:
            r.fase(f"{self._locale.topic}_chiamata")

    def pub_fine(self, r, topic):
        r.fase(f"{topic}_fine")
        self._locale.r = None

    def fine_pubblicazione(self, r, esito="pubblicata"):
        self._locale.r = None
        r.dati.setdefault("esito", esito)
        self._fine_parte(r)

    def _fine_parte(self, r):
        with self._lock:
            r.parti -= 1
            if r.parti > 0:
                return
        d = r.dati
        riga = [r.seq, d.get("frame_idx"), d.get("stamp_in"), d.get("n_punti"), d.get("esito"),
                d.get("tid_cb"), d.get("tid_pub"), d.get("dds_src_ns"), d.get("dds_recv_ns"),
                d.get("pub_seq"), d.get("ric_seq"), d.get("ritardo_ricezione_ms"), d.get("ritardo_dds_ms"), d.get("durata_cb_ms"),
                r.stack_attesa, r.stack_durata]
        for e in EVENTI:
            m_w = r.t.get(e)
            riga += list(m_w) if m_w is not None else [None, None]
        self._scr.add("righe", riga)

    # --- garbage collector ------------------------------------------------------

    def _gc_cb(self, fase, info):
        if fase == "start":
            self._gc_t0 = _adesso()
            return
        t0, self._gc_t0 = self._gc_t0, None
        m, w = _adesso()
        if t0 is None:
            return
        self._scr.add("gc", (t0[0], t0[1], m, w, (m - t0[0]) / 1e6, info.get("generation"),
                             info.get("collected"), info.get("uncollectable"), threading.get_native_id()))

    # --- CPU sampler (own thread) ----------------------------------------------

    @staticmethod
    def _leggi(p):
        try:
            with open(p, "rb") as f:
                return f.read().decode()
        except OSError:
            return ""

    def campione_cpu(self):
        m, w = _adesso()
        st = self._leggi("/proc/stat")
        cpu = [None] * 8
        run = blk = None
        for riga in st.splitlines():
            if riga.startswith("cpu "):
                v = riga.split()[1:9]
                cpu = [int(x) for x in v] + [None] * (8 - len(v))
            elif riga.startswith("procs_running"):
                run = int(riga.split()[1])
            elif riga.startswith("procs_blocked"):
                blk = int(riga.split()[1])
        la = self._leggi("/proc/loadavg").split()
        load = [float(x) for x in la[:3]] if len(la) >= 4 else [None] * 3
        rq = la[3] if len(la) >= 4 else None
        some = full = None
        for riga in self._leggi("/proc/pressure/cpu").splitlines():
            campi = dict(c.split("=", 1) for c in riga.split()[1:] if "=" in c)
            if riga.startswith("some"):
                some = campi.get("total")
            elif riga.startswith("full"):
                full = campi.get("total")
        ps = self._leggi("/proc/self/stat")
        ut = stt = nth = None
        if ")" in ps:
            # fields after the command name: state is 3, utime 14, stime 15, threads 20
            v = ps.rsplit(")", 1)[1].split()
            if len(v) > 17:
                ut, stt, nth = v[11], v[12], v[17]
        self._scr.add("cpu", [m, w] + cpu + [run, blk] + load + [rq, some, full, ut, stt, nth])

    def _cpu_loop(self):
        while not self._stop.wait(self._periodo_cpu):
            try:
                self.campione_cpu()
            except Exception:
                pass

    # --- shutdown ---------------------------------------------------------------

    def chiudi(self):
        if self._chiuso:
            return
        self._chiuso = True
        self._stop.set()
        try:
            gc.callbacks.remove(self._gc_cb)
        except ValueError:
            pass
        self.cane.chiudi()
        self._t_cpu.join(timeout=2.0)
        self._scr.chiudi()
