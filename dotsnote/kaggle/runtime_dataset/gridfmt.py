"""Compact ARC grid format for prompts and answers ("Compact grid A/B").

json (default, unchanged): the ARC Prize benchmark prompt, grids as json.dumps ("[[0, 0, 1], [3, 4, 5]]",
  ~3 tokens/cell with the lb144r tokenizer).
compact: one row per line, one digit per cell, no separators, with a "(HxW)" shape header ("INPUT (2x3):\n001\n345").
  Digits are single tokens and never merge, so a row costs W + 1 tokens (newline). Adds FORMAT_NOTE (how to read the
  grids), WRITE_NOTE (write grids compactly in the reasoning, refer to known rows/grids by name instead of copying)
  and ANSWER_NOTE (final answer as the usual JSON list of rows: lb144r's trained answer form; a pilot that asked for a
  compact final answer got hybrid JSON lists of multi-digit row numbers, which lose leading zeros).
compact_noinstr: compact prompt grids + FORMAT_NOTE + ANSWER_NOTE (no instruction on how to write the reasoning).

The answer parser (parse_answer) accepts both forms: a valid JSON grid of 0-9 cells first (the caller's existing JSON
parser), else a JSON list whose leaves are multi-digit row numbers/strings (one leaf per row, left-padded with zeros to
the widest row), else the last block of consecutive compact rows ("0012", or single digits separated by one
space/comma) of equal width.
"""
import json
import re

FORMATS = ("json", "compact", "compact_noinstr")

FORMAT_NOTE = ("Grid format: each grid is written one row per line, one digit (0-9) per cell, with no separators; "
               "(HxW) gives its height (rows) x width (columns).")
WRITE_NOTE = ("Write every grid or row in your reasoning in this same compact form (e.g. 0012), never as "
              "comma-separated lists. Do not copy rows or grids that are already written above: refer to them by name "
              "(e.g. \"Example 1 input row 3\", \"test input rows 2-5\"); for a row that differs from a known row, name "
              "that row and list only the changed cells (e.g. \"= test input row 4 except col 7 -> 2\").")
ANSWER_NOTE = "Give the final output grid as a JSON list of rows, e.g. [[0, 0, 1], [3, 4, 5]]."
ANCHOR = "\n\n--Training Examples--"


def grid_compact(g):
    return "\n".join("".join(str(v) for v in r) for r in g)


def shape_tag(g):
    return f"({len(g)}x{len(g[0]) if g else 0})"


def training_examples(task, fmt):
    ex = ""
    if fmt == "json":
        for i, p in enumerate(task["train"]):
            ex += f"--Example {i}-- \n\n INPUT: \n\n" + json.dumps(p["input"]) + "\n\n"
            ex += "OUTPUT: \n\n" + json.dumps(p["output"]) + "\n\n"
        return ex
    for i, p in enumerate(task["train"]):
        ex += f"--Example {i}--\nINPUT {shape_tag(p['input'])}:\n{grid_compact(p['input'])}\n"
        ex += f"OUTPUT {shape_tag(p['output'])}:\n{grid_compact(p['output'])}\n\n"
    return ex


def prompt_body(template, task, ti, fmt="json"):
    """The user message: template filled with the task's grids in `fmt` (json == the benchmark prompt, byte-exact)."""
    tin = task["test"][ti]["input"]
    if fmt == "json":
        return template.format(training_examples=training_examples(task, fmt), test_input=json.dumps(tin))
    if fmt not in FORMATS:
        raise ValueError(fmt)
    body = template.format(training_examples=training_examples(task, fmt),
                           test_input=f"INPUT {shape_tag(tin)}:\n{grid_compact(tin)}")
    note = FORMAT_NOTE + (" " + WRITE_NOTE if fmt == "compact" else "") + " " + ANSWER_NOTE
    assert ANCHOR in body
    return body.replace(ANCHOR, "\n\n" + note + ANCHOR, 1)


# ---------------------------------------------------------------- parsing
_ROW_TIGHT = re.compile(r"\d+")
_ROW_SEP = re.compile(r"\d(?:[ ,]\d)+")


def _row(line):
    s = line.strip().strip("`|").strip()
    if _ROW_TIGHT.fullmatch(s) or _ROW_SEP.fullmatch(s):
        return [int(c) for c in s if c.isdigit()]
    return None


def compact_blocks(text):
    """[(start_char, end_char, grid)] for every maximal block of consecutive compact-row lines of equal width
    (<= 30 x 30)."""
    out, cur, pos, start = [], [], 0, 0
    lines = text.split("\n")

    def flush(end):
        if cur and len(cur) <= 30 and len(cur[0]) <= 30:
            out.append((start, end, [list(r) for r in cur]))

    for line in lines:
        r = _row(line)
        if r is not None and cur and len(r) != len(cur[0]):
            flush(pos - 1); cur = []
        if r is not None:
            if not cur:
                start = pos
            cur.append(r)
        elif cur:
            flush(pos - 1); cur = []
        pos += len(line) + 1
    if cur:
        flush(pos - 1)
    return out


def parse_compact(text):
    b = compact_blocks(text)
    return b[-1][2] if b else None


def is_grid(g):
    return isinstance(g, list) and bool(g) and all(isinstance(r, list) and r and all(
        isinstance(v, int) and not isinstance(v, bool) for v in r) for r in g)


def _leaves(x):
    if isinstance(x, list):
        for y in x:
            yield from _leaves(y)
    else:
        yield x


def rows_from_hybrid(g):
    """[[1110, 2221], ...] / [["0012", "3456"]]: a JSON list whose leaves are whole rows -> grid (None if not that)."""
    if not isinstance(g, list):
        return None
    lv = list(_leaves(g))
    if not lv or not all((isinstance(v, int) and not isinstance(v, bool) and v >= 0) or
                         (isinstance(v, str) and v.isdigit()) for v in lv):
        return None
    if not any(isinstance(v, str) or v > 9 for v in lv):
        return None
    rows = [str(v) for v in lv]
    w = max(len(r) for r in rows)
    if w > 30 or len(rows) > 30:
        return None
    return [[int(c) for c in r.zfill(w)] for r in rows]


def parse_answer(answer, json_parse):
    """Old form first (json_parse(answer) must give a list of 0-9 int lists), else a hybrid JSON list of row numbers,
    else the last compact block."""
    try:
        g = json_parse(answer)
    except Exception:
        g = None
    if is_grid(g) and all(0 <= v <= 9 for r in g for v in r):
        return g
    h = rows_from_hybrid(g)
    if h is not None:
        return h
    c = parse_compact(answer)
    return c if c is not None else (g if is_grid(g) else None)


def trace_grid_compact(text, shape, test_in):
    """Last compact block in a reasoning trace of the expected shape that is not the test input (cf. solve.trace_grid)."""
    for _, _, g in reversed(compact_blocks(text)):
        if g == test_in:
            continue
        if shape is None or (len(g), len(g[0])) == tuple(shape):
            return g
    return None
