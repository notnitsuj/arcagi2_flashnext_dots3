"""Confidence-gated termination of reasoning traces. Shipped next to solve.py like loop_detect.py.

DeepConf-style online signal (arXiv 2508.15260): token confidence c_i = -mean of the top-k logprobs at position i
(k = the request's `logprobs`), group confidence = mean c over a sliding window of `group` tokens. At each checkpoint
(completion-token count) the gate compares the chosen feature of the trace so far with that checkpoint's threshold and
says "kill" when the trace looks doomed (feature below threshold: low confidence).

Features (all over the tokens seen so far):
  lowest_group    min sliding-window group confidence (window `group`, evaluated every `stride` tokens)
  bottom10_group  mean of the lowest 10% of those group confidences
  avg_conf        mean token confidence
  tail_conf       mean token confidence of the last `group` tokens
  neg_nll         mean logprob of the sampled token

Configuration string (env ARC_CONF_KILL in solve.py), default off:
  "<x>:<thr>[,<x>:<thr>...]"   e.g. "100000:7.9"  (checkpoints in completion tokens, thresholds in feature units)
plus ARC_CONF_FEATURE (default lowest_group), ARC_CONF_K (top-k logprobs requested, default 5), ARC_CONF_GROUP (2048).
"""
import math
import os
from collections import deque

FEATURES = ("lowest_group", "bottom10_group", "avg_conf", "tail_conf", "neg_nll")


def parse_spec(spec):
    """'x:thr,x:thr' -> sorted [(x, thr)]; '' / None -> []"""
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        x, thr = part.split(":")
        out.append((int(float(x)), float(thr)))
    return sorted(out)


class ConfGate:
    def __init__(self, checkpoints, feature="lowest_group", k=5, group=2048, stride=256):
        if feature not in FEATURES:
            raise ValueError(f"unknown feature {feature}")
        self.checkpoints = list(checkpoints)
        self.feature, self.k, self.group, self.stride = feature, k, group, stride
        self.n = 0
        self.sum_c = 0.0
        self.sum_lp = 0.0
        self.win = deque()
        self.win_sum = 0.0
        self.groups = []           # group means sampled every `stride` tokens once the window is full
        self.next_cp = 0
        self.killed_at = None
        self.last_value = None

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        cps = parse_spec(env.get("ARC_CONF_KILL", ""))
        if not cps:
            return None
        return cls(cps, feature=env.get("ARC_CONF_FEATURE", "lowest_group"), k=int(env.get("ARC_CONF_K", "5")),
                   group=int(env.get("ARC_CONF_GROUP", "2048")))

    def clone(self):
        return ConfGate(self.checkpoints, self.feature, self.k, self.group, self.stride)

    # ------------------------------------------------------------------ input
    def feed_token(self, token_logprob, top_logprobs):
        """one generated token: its logprob and the top-k logprob values (any iterable of floats)"""
        tops = sorted(top_logprobs, reverse=True)[:self.k]
        c = -sum(tops) / len(tops) if tops else 0.0
        if token_logprob is None or (isinstance(token_logprob, float) and math.isnan(token_logprob)):
            token_logprob = tops[0] if tops else 0.0
        self.n += 1
        self.sum_c += c
        self.sum_lp += token_logprob
        self.win.append(c); self.win_sum += c
        if len(self.win) > self.group:
            self.win_sum -= self.win.popleft()
        if len(self.win) == self.group and self.n % self.stride == 0:
            self.groups.append(self.win_sum / self.group)

    def feed_openai(self, lp):
        """the `logprobs` object of one streamed /v1/completions choice; returns True if the gate fires now"""
        if not lp or not lp.get("tokens"):
            return False
        tops = lp.get("top_logprobs") or [None] * len(lp["tokens"])
        fire = False
        for tl, tp in zip(lp.get("token_logprobs") or [None] * len(lp["tokens"]), tops):
            self.feed_token(tl, (tp or {}).values())
            fire = self._check() or fire
        return fire

    # ------------------------------------------------------------------ decision
    def value(self):
        if self.n == 0:
            return None
        if self.feature == "avg_conf":
            return self.sum_c / self.n
        if self.feature == "neg_nll":
            return self.sum_lp / self.n
        if self.feature == "tail_conf":
            return self.win_sum / len(self.win)
        g = self.groups or [self.win_sum / len(self.win)]
        if self.feature == "lowest_group":
            return min(g)
        q = sorted(g)
        m = max(1, len(q) // 10)
        return sum(q[:m]) / m

    def _check(self):
        if self.killed_at is not None or self.next_cp >= len(self.checkpoints):
            return False
        x, thr = self.checkpoints[self.next_cp]
        if self.n < x:
            return False
        self.next_cp += 1
        v = self.value()
        self.last_value = v
        if v is not None and v < thr:
            self.killed_at = self.n
            return True
        return False

    @property
    def done(self):
        """no checkpoints left: the caller may stop paying for logprobs (not possible mid-request today)"""
        return self.next_cp >= len(self.checkpoints)
