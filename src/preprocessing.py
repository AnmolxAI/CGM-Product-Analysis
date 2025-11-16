"""Preprocessing utilities for the DCGM consumer posts project.

This module is a placeholder to gradually refactor data loading and cleaning
logic out of analysis.py into reusable functions.
"""

import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load the CGM dataset from an Excel file.

    Parameters
    ----------
    path : str
        Path to the Excel file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    return pd.read_excel(path)
