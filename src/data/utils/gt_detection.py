from __future__ import annotations
import pickle
from pathlib import Path
from typing import Iterable, List, Tuple
import numbers

try:
    import numpy as np                    # optional, but convenient
except ModuleNotFoundError:               # fall back gracefully
    np = None

# --------------------------------------------------------------------------- #
def get_gt_intervals(
    anom_type: str,
    sample_id: int,
    split: str = "train",
    *,
    root: Path = Path("all_data/synthetic"),
) -> List[Tuple[int, int]]:
    """
    Return ground-truth anomaly intervals for one time-series sample.

    Parameters
    ----------
    anom_type : str
        Folder name under ``data/synthetic/`` (e.g. ``"point"``, ``"range"``).
    sample_id : int
        1-based index of the series you want.
    split : {"train", "eval", "val", "test"}, default "train"
        Which dataset split to use (aliases map to the existing folders).
    root : pathlib.Path, optional
        Base directory that holds all synthetic data (default ``data/synthetic``).

    Returns
    -------
    List[Tuple[int, int]]
        A list of (start, end) index pairs, with ``start`` and ``end`` inclusive.
    """
    split = _norm_split(split)                        # normalise names
    pkl_path = root / anom_type / split / "data.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"❌  Data file not found: {pkl_path}")

    # 1. Load pickle (⚠️ only do this with files you trust)
    with pkl_path.open("rb") as fh:
        data = pickle.load(fh)

    if not (isinstance(data, dict)
            and isinstance(data.get("series"), list)
            and isinstance(data.get("anom"),   list)):
        raise ValueError("❌  Pickle does not have the expected structure "
                         "with top-level keys 'series' and 'anom' (both lists).")

    series, anom = data["series"], data["anom"]
    n_pairs = min(len(series), len(anom))
    idx = sample_id - 1
    if not (0 <= idx < n_pairs):
        raise IndexError(f"ID must be between 1 and {n_pairs}")

    return _to_intervals(anom[idx])


# --------------------------------------------------------------------------- #
# helpers
def _norm_split(name: str) -> str:
    """Map common aliases to the two actual folders ('train', 'eval')."""
    n = name.lower()
    if n in {"train", "tr"}:
        return "train"
    if n in {"eval", "val", "validation", "test"}:
        return "eval"
    raise ValueError(f"Unknown split: {name!r}")


def _to_intervals(obj) -> List[Tuple[int, int]]:
    """
    Convert various encodings of anomaly positions to a list of (start, end).
    Handles:
        • list/tuple/ndarray of pairs
        • flat list/array with 2*k elements  -> (x0,x1), (x2,x3), …
        • list/array of single indices      -> (i,i) for point anomalies
        • nested one-element wrapper like [[...]]
    """
    # unwrap single-element wrappers (e.g. [[(a,b), (c,d)]])
    if isinstance(obj, (list, tuple)) and len(obj) == 1 and not _is_pair(obj[0]):
        return _to_intervals(obj[0])

    # NumPy array support
    if np is not None and isinstance(obj, np.ndarray):
        arr = obj.astype(int)
        if arr.ndim == 2 and arr.shape[1] == 2:          # explicit pairs
            return [(int(s), int(e)) for s, e in arr]
        arr = arr.ravel()                                # flat otherwise
        return _flat_to_pairs(arr.tolist())

    # Pure Python sequences
    if isinstance(obj, (list, tuple)):
        if all(_is_pair(el) for el in obj):              # [(s,e), ...]
            return [(int(s), int(e)) for s, e in obj]
        if all(isinstance(el, numbers.Real) for el in obj):
            return _flat_to_pairs(list(map(int, map(float, obj))))
        # fall through: unsupported nested structure
    raise TypeError(f"Cannot interpret anomaly structure of type {type(obj).__name__}")


def _is_pair(x) -> bool:
    """True if x looks like a 2-element pair of reals."""
    return (isinstance(x, (list, tuple)) and len(x) == 2
            and all(isinstance(v, numbers.Real) for v in x))


def _flat_to_pairs(seq: Iterable[int]) -> List[Tuple[int, int]]:
    """Turn [a,b,c,d] → [(a,b),(c,d)]. If odd length, treat each as a point."""
    seq = list(seq)
    if len(seq) % 2 == 0 and len(seq) >= 2:
        return [(seq[i], seq[i + 1]) for i in range(0, len(seq), 2)]
    # interpret each element as a point anomaly
    return [(v, v) for v in seq]

# --------------------------------------------------------------------------- #
# Quick sanity check (remove or wrap in a test if you prefer)
if __name__ == "__main__":
    example = get_gt_intervals("point", 69, split="eval")
    print("Example intervals:", example)
    example = get_gt_intervals("noisy-point", 69, split="eval")
    print("Example intervals:", example)
    example = get_gt_intervals("trend", 169, split="eval")
    print("Example intervals:", example)
    example = get_gt_intervals("noisy-freq", 69, split="eval")
    print("Example intervals:", example)
    example = get_gt_intervals("freq", 69, split="eval")
    print("Example intervals:", example)
