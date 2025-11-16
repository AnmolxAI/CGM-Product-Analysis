"""Text processing helpers for the DCGM consumer posts project.

Over time, move tokenization, stopword removal, lemmatization and n-gram
generation logic from analysis.py into reusable functions here.
"""

import re
from typing import Iterable, List

def to_lower(text: str) -> str:
    return text.lower() if isinstance(text, str) else text

def remove_urls(text: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(r"http\S+|www\S+", "", text)
