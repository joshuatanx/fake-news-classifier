import pandas as pd
from typing import Final

# Column names
TITLE_COL: Final[str] = "title"
TEXT_COL: Final[str] = "text"
LABEL_COL: Final[str] = "label"
CONTENT_COL: Final[str] = "content"

# Label definitions
LABEL_REAL: Final[int] = 1
LABEL_FAKE: Final[int] = 0

LABEL_MAP: Final[dict[int, str]] = {
    LABEL_REAL: "real",
    LABEL_FAKE: "fake"
}

INVERSE_LABEL_MAP: Final[dict[str, int]] = {
    "real": LABEL_REAL,
    "fake": LABEL_FAKE
}

# Schema validation
REQUIRED_COLUMNS: Final[list[str]] = [TEXT_COL, LABEL_COL]
OPTIONAL_COLUMNS: Final[list[str]] = [TITLE_COL]
DERIVED_COLUMNS: Final[list[str]] = [CONTENT_COL]

INVALID_TYPES = (list, dict, set, tuple, pd.Series, pd.DataFrame)
def is_string_castable(x):
    """Return whether `x` can be cast to a string."""
    if isinstance(x, INVALID_TYPES):
        return False
    
    if pd.isna(x):
        return True
    
    try:
        str(x)
        return True
    except Exception:
        return False

def validate_raw_dataframe(df: pd.DataFrame):
    """Validate that a raw dataframe has the required columns with valid entries.
    
    Raises:
        `TypeError`: If a text column contains values that cannot be converted to strings.
        `ValueError`: If a required column is missing or labels are invalid.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column: {missing}")
    
    text_columns = [TEXT_COL] + [column for column in OPTIONAL_COLUMNS if column in df.columns]
    for column in text_columns:
        uncastable_mask = ~df[column].apply(is_string_castable)
        
        if uncastable_mask.any():
            uncastable_indices = df.index[uncastable_mask].tolist()
            raise TypeError(f"Column \"{column}\" contains value(s) that cannot be converted to a string at row(s) {uncastable_indices}")
    
def validate_dataframe(df: pd.DataFrame):
    """Validate that a dataframe follows the expected schema.
    
    Raises:
        `TypeError`: If a text column contains invalid entries.
        `ValueError`: If dataframe schema is invalid.
    """
    validate_raw_dataframe(df)

    text_columns = [TEXT_COL] + [column for column in OPTIONAL_COLUMNS if column in df.columns]
    for column in text_columns:
        missing_mask = df[column].isna()
        
        if missing_mask.any():
            missing_indices = df.index[missing_mask].tolist()
            raise ValueError(f"Column \"{column}\" contains missing value(s) at row(s) {missing_indices}")
    
    if not df[LABEL_COL].isin(LABEL_MAP.keys()).all():
        raise ValueError(f"Invalid labels found. Expected one of {list(LABEL_MAP.keys())}")