"""Streaming detector for degenerate repetition in a reasoning trace (stdlib only, solve.py-compatible).

A trace is flagged when the zlib compression ratio of its last `window` characters stays below `thr` over `persist`
consecutive characters of new text (checked every `stride` characters). Normal ARC reasoning compresses to ~0.2-0.45
(grids and step-by-step cell checks are the most compressible legit text, min 0.127 over 202 correct traces); runaway
loops ("✓✓✓...", "7 7 7 ...", a repeated sentence) compress to 0.002-0.06.

    det = LoopDetector()
    for chunk in stream:
        text += chunk
        if det.feed(chunk):
            ...  # abort the request, then force a conclusion on `text`

Calibration (python dots3/loop_detect.py): replays every runs/*_text.jsonl trace through the detector and reports
false positives on traces that ended correct, and where it fires on capped (finish=length) traces.
"""
import zlib

__all__ = ["LoopDetector", "first_fire"]


class LoopDetector:
    def __init__(self, window=8192, stride=1024, thr=0.10, persist=49152, min_chars=8192):
        self.window, self.stride, self.thr, self.persist, self.min_chars = window, stride, thr, persist, min_chars
        self.buf = ""          # last `window` chars
        self.n = 0             # total chars seen
        self.next_check = max(window, min_chars)
        self.low_since = None  # char position where the current low-ratio run started
        self.fired_at = None
        self.last_ratio = None
        self.loop_start = None

    def ratio(self):
        b = self.buf.encode("utf-8", "replace")
        return len(zlib.compress(b, 6)) / max(1, len(b))

    def feed(self, chunk):
        """Add streamed text; returns True once the trace is judged degenerate (stays True afterwards)."""
        if self.fired_at is not None:
            return True
        self.buf = (self.buf + chunk)[-self.window:]
        self.n += len(chunk)
        while self.n >= self.next_check:      # cheap: one compress of <=8 KiB per `stride` chars
            self.next_check += self.stride
            r = self.last_ratio = self.ratio()
            if r < self.thr:
                if self.low_since is None:
                    self.low_since = self.n
                if self.n - self.low_since >= self.persist:
                    self.fired_at = self.n
                    # approximate start of the degenerate span: strip text[loop_start:] before forcing a conclusion
                    self.loop_start = max(0, self.low_since - self.window)
                    return True
            else:
                self.low_since = None
        return False


def first_fire(text, chunk=256, **kw):
    """Replay a full trace through the detector in streaming-sized chunks; char position of firing or None."""
    det = LoopDetector(**kw)
    for i in range(0, len(text), chunk):
        if det.feed(text[i:i + chunk]):
            return det.fired_at
    return None


if __name__ == "__main__":
    import argparse, glob, json, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))
    ap.add_argument("--grid", action="store_true", help="sweep thr x persist")
    a = ap.parse_args()
    traces = []
    for tf in sorted(glob.glob(os.path.join(a.runs, "*_text.jsonl"))):
        mf = tf.replace("_text.jsonl", ".jsonl")
        if not os.path.exists(mf):
            continue
        meta = {(m["task"], m["test"], m.get("attempt", 0)): m for m in map(json.loads, open(mf))}
        for line in open(tf):
            r = json.loads(line)
            m = meta.get((r["task"], r["test"], r.get("attempt", 0)))
            if m is not None:
                traces.append((os.path.basename(tf)[:-11], r["task"], r["test"], m, r["completion"]))
    lb = lambda run: run.startswith("hold_lb144")
    settings = [(t, p) for t in (0.05, 0.08, 0.10, 0.15) for p in (0, 8192, 16384, 32768, 49152)] if a.grid \
        else [(0.10, 49152)]
    print(f"{len(traces)} traces: {sum(bool(m['correct']) for *_, m, _ in traces)} correct, "
          f"{sum(m['finish'] == 'length' for *_, m, _ in traces)} capped")
    print(f"{'thr':>5} {'persist':>7} | {'FP corr':>7} {'FP stop-wrong':>13} | {'fire/capped all':>15} {'saved tok':>9} |"
          f" {'fire/capped lb144':>17} {'saved tok':>9}")
    for thr, per in settings:
        fp_c = fp_w = 0; fire = [0, 0]; capn = [0, 0]; saved = [0, 0]; rows = []
        for run, task, test, m, txt in traces:
            pos = first_fire(txt, thr=thr, persist=per)
            tok = None if pos is None else int(pos / max(1, len(txt)) * m["ct"])
            if m["correct"]:
                fp_c += pos is not None
                if pos is not None and len(settings) == 1:
                    rows.append(f"  FALSE POSITIVE (correct trace): {run} {task}/{test} at tok~{tok}/{m['ct']}")
            elif m["finish"] == "stop":
                fp_w += pos is not None
            if m["finish"] == "length":
                for j, on in enumerate((True, lb(run))):
                    if on:
                        capn[j] += 1
                        if pos is not None:
                            fire[j] += 1; saved[j] += m["ct"] - tok
                if len(settings) == 1:
                    rows.append(f"  capped {run:18s} {task}/{test}: " +
                                (f"fires at tok~{tok // 1000}k of {m['ct'] // 1000}k (saves {(m['ct'] - tok) // 1000}k)"
                                 if pos is not None else "no fire"))
        print(f"{thr:5.2f} {per:7d} | {fp_c:7d} {fp_w:13d} | {fire[0]:7d}/{capn[0]:<7d} {saved[0]:9d} |"
              f" {fire[1]:9d}/{capn[1]:<7d} {saved[1]:9d}")
        for r in rows:
            print(r)
