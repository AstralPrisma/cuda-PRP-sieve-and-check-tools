"""Strict, dependency-free interchange for one-candidate Catena tasks.

The expression is a description, never executable Python. ``terms`` is the
number of factors, including the factor at ``start``; ``step`` is 1 or 2.
"""
from __future__ import annotations

from pathlib import Path
import re

MAX_U64 = (1 << 64) - 1
MAX_TERMS = 1_000_000
TASK_FORMAT_VERSION = 1
_DECIMAL = r"[1-9][0-9]{0,19}"
_TASK_RE = re.compile(
    rf"CSPS\nCatena\(x=({_DECIMAL}),terms=({_DECIMAL}),step=([12])\)([+-])1\n"
)


def validate_parameters(start: int, terms: int, step: int, sign: int) -> dict[str, int]:
    """Validate arithmetic bounds without constructing the potentially huge N."""
    values = {"start": start, "terms": terms, "step": step, "sign": sign}
    if any(type(value) is not int for value in values.values()):
        raise ValueError("task parameters must be integers (not bool or float)")
    if not 1 <= start < MAX_U64:
        raise ValueError("start must be positive and start + 1 must fit uint64")
    if not 1 <= terms <= MAX_TERMS:
        raise ValueError("terms must be 1..1000000")
    if step not in (1, 2):
        raise ValueError("step must be 1 or 2")
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    if start + step * (terms - 1) >= MAX_U64:
        raise ValueError("last base + 1 must fit uint64")
    # This is the only N < 2 case for positive starts/terms and these steps.
    if start == 1 and terms == 1 and sign == -1:
        raise ValueError("candidate is less than 2")
    return values


def expression(start: int, terms: int, step: int, sign: int) -> str:
    validate_parameters(start, terms, step, sign)
    return f"Catena(x={start},terms={terms},step={step}){'+' if sign == 1 else '-'}1"


def task_filename(start: int, terms: int, step: int, sign: int) -> str:
    validate_parameters(start, terms, step, sign)
    return f"CSPS_d{step}_m{terms}_x{start}_{'plus' if sign == 1 else 'minus'}.txt"


def make_task(start: int, terms: int, step: int, sign: int) -> str:
    """Return exactly two LF-terminated UTF-8/ASCII lines."""
    return "CSPS\n" + expression(start, terms, step, sign) + "\n"


def parse_task(data: str | bytes, *, filename: str | Path | None = None) -> dict[str, int]:
    """Parse one canonical task; optionally verify its canonical filename.

    A BOM, CRLF, whitespace, extra lines, noncanonical integers and expression
    aliases are deliberately rejected, so a task has just one representation.
    """
    if isinstance(data, bytes):
        if len(data) > 256:
            raise ValueError("task is too large")
        try:
            data = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("task must be UTF-8") from exc
    if not isinstance(data, str) or len(data) > 256:
        raise ValueError("task must be a short UTF-8 string")
    match = _TASK_RE.fullmatch(data)
    if match is None:
        raise ValueError("expected CSPS and one canonical Catena expression on two LF lines")
    start, terms, step = map(int, match.group(1, 2, 3))
    sign = 1 if match.group(4) == "+" else -1
    result = validate_parameters(start, terms, step, sign)
    if filename is not None:
        expected = task_filename(**result)
        if Path(filename).name != expected:
            raise ValueError(f"task filename does not match its parameters; expected {expected}")
    return result


def read_task(path: str | Path) -> dict[str, int]:
    path = Path(path)
    with path.open("rb") as stream:
        data = stream.read(257)
    return parse_task(data, filename=path.name)
