# api/preprocess.py
import re
from typing import Any, List
import numpy as np
import pandas as pd

_URL_RE = re.compile(r"http\S+|www\.\S+", flags=re.IGNORECASE)
_NONALNUM_RE = re.compile(r"[^a-z0-9\s]+")

def _clean_one(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.lower()
    s = _URL_RE.sub(" ", s)
    s = _NONALNUM_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def basic_clean(x: Any) -> List[str]:
    """
    Always return an iterable of strings with the SAME number of items received.
    - If x is Series/ndarray/list/tuple -> return list[str] of same length
    - If x is a single value/string -> return [str]
    This keeps sklearn's pipeline happy.
    """
    if isinstance(x, pd.Series):
        return x.astype(str).apply(_clean_one).tolist()
    if isinstance(x, (np.ndarray, list, tuple)):
        return [_clean_one(v) for v in x]
    return [_clean_one(x)]
