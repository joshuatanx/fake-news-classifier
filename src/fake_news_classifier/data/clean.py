import math
import pandas as pd
import pycld2 as cld2
import re
from .schema import validate_raw_dataframe, LABEL_COL, TITLE_COL, TEXT_COL

def cast_text_columns_to_string(df: pd.DataFrame):
    """Cast text columns to string dtype while preserving `pd.NA`."""
    df = df.copy()
    
    text_columns = [TITLE_COL, TEXT_COL] if TITLE_COL in df.columns else [TEXT_COL]
    
    def to_text(x):
        if pd.isna(x):
            return pd.NA

        if isinstance(x, float) and math.isfinite(x) and x.is_integer():
            return str(int(x))

        return str(x)
    
    for column in text_columns:
        df[column] = df[column].map(to_text).astype("object")
    
    return df

def drop_invalid_label_rows(df: pd.DataFrame):
    """Drop rows with labels that are not `0` or `1`."""
    df = df.copy()
    
    invalid_mask = df[LABEL_COL].apply(lambda label: label not in [0, 1])
    
    if invalid_mask.any():
        invalid_indices = df.index[invalid_mask].tolist()
        df.drop(index = invalid_indices, inplace = True)
    
    return df

def replace_whitespace_entries(df: pd.DataFrame):
    """Replace entries consisting of solely whitespace with `pd.NA`."""
    df = df.copy()
    
    df.replace(r"^\s*$", pd.NA, regex = True, inplace = True)
    
    return df

def drop_missing_text_rows(df: pd.DataFrame):
    """Drop rows missing text."""
    df = df.copy()
    
    df.dropna(subset = [TEXT_COL], inplace = True)

    return df

def fill_na_with_empty_strings(df: pd.DataFrame):
    """Fill `pd.NA` values with empty strings."""
    return df.fillna("")

PARAGRAPH_TOKEN = " __PARAGRAPH__ "
def protect_paragraphs(text: str, n_newlines: int = 1, paragraph_token: str = PARAGRAPH_TOKEN):
    """Find all paragraphs separated by at least `n_newlines` newlines and replace the space with `paragraph_token`."""
    PARA_SPLIT_RE = re.compile(rf"(?:\r?\n\s*){{{n_newlines},}}")
    return PARA_SPLIT_RE.sub(paragraph_token, text)

def remove_ansi_codes(text: str):
    """Replace ANSI escape codes with whitespace."""
    return re.sub(r"\x1B(?:[@-Z\\-_]|[78]|\[[0-?]*[ -/]*[@-~])", " ", text)

def remove_control_codes(text: str):
    """Replace spacing codes with whitespace and remove the other C0 control codes, DEL, C1 control codes, BOM, and zero-width characters."""
    translation_map = {}
    
    # Spacing codes
    whitespace_codes = [
        0x09,   # Tab
        0x0A,   # Newline
        0x0D,   # Carriage
        0x0B,   # Vertical tab
        0x0C    # Form feed
    ]
    for code in whitespace_codes:
        translation_map[code] = 0x20    # Space
    
    # C0 controls excluding whitespace codes
    for code in range(0x00, 0x20):
        if code not in whitespace_codes:
            translation_map[code] = None
    
    # DEL
    translation_map[0x7F] = None

    # C1 controls (which can result from poor decoding)
    for code in range(0x80, 0xA0):
        translation_map[code] = None

    # BOM + zero-width chars
    for code in (0xFEFF, 0x200B, 0x200C, 0x200D, 0x200E, 0x200F):
        translation_map[code] = None
    
    return text.translate(translation_map)

def fix_missing_spaces_around_punctuation(text: str):
    """Fix missing spaces before/after punctuation."""
    URL_RE = re.compile(r"https?://\S+|www\.\S+")
    EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")

    # Protect URLs and emails
    protected = []
    def _protect(match):
        protected.append(match.group(0))
        return f"__PROTECTED_{len(protected) - 1}__"

    text = URL_RE.sub(_protect, text)
    text = EMAIL_RE.sub(_protect, text)

    # Split after acronym chains, e.g. U.S.President -> U.S. President
    text = re.sub(
        r"((?:\b[A-Z]\.){2,})([A-Za-z])",
        r"\1 \2",
        text
    )
    
    # Fix missing space before `(`, `[`, and `{` when preceeded by an alphanumeric value
    text = re.sub(r"(?<=[A-Za-z0-9])(?=[([{])", " ", text)

    # Fix missing space after ')', ']', and '}', e.g. (Reuters)WASHINGTON -> (Reuters) WASHINGTON
    text = re.sub(
        r"([\)\]\}])(?=[A-Za-z])",
        r"\1 ",
        text
    )

    # Fix missing space after '.', '!', and '?' when the next sentence starts with a capital, `(`, `[`, or `{`
    text = re.sub(
        r"([.!?])(?![A-Z]\.)(?=[A-Z([{])",
        r"\1 ",
        text
    )

    # Restore protected spans
    for i, span in enumerate(protected):
        text = text.replace(f"__PROTECTED_{i}__", span)

    return text

def normalize_whitespace(text: str):
    """Remove repeated whitespace."""
    return re.sub(r"\s+", " ", text).strip()

def unprotect_paragraphs(text: str, paragraph_token: str = PARAGRAPH_TOKEN):
    """Replace all instances of `paragraph_token` with \"\\n\\n\"."""
    return text.replace(paragraph_token, "\n\n")

def apply_regex_substitutions(text: str, substitutions: dict[str, str]):
    """Apply ordered regex substitutions to text."""
    for pattern, replacement in substitutions.items():
        text = re.sub(pattern, replacement, text)
    
    return text

def canonicalize_text(
    text: str,
    regex_substitutions: dict[str, str] | None = None,
    paragraph_newlines: int = 1,
    paragraph_token: str = PARAGRAPH_TOKEN
):
    """Remove any invalid characters, fix whitespace, and apply RegEx substitutions."""
    text = protect_paragraphs(text, n_newlines = paragraph_newlines, paragraph_token = paragraph_token)
    text = remove_ansi_codes(text)
    text = remove_control_codes(text)
    text = fix_missing_spaces_around_punctuation(text)
    text = normalize_whitespace(text)
    text = unprotect_paragraphs(text, paragraph_token = paragraph_token)
    
    if regex_substitutions:
        text = apply_regex_substitutions(text, substitutions = regex_substitutions)
        text = fix_missing_spaces_around_punctuation(text)
        
        text = protect_paragraphs(text, n_newlines = paragraph_newlines, paragraph_token = paragraph_token)
        text = normalize_whitespace(text)
        text = unprotect_paragraphs(text, paragraph_token = paragraph_token)
    
    return text

def canonicalize_text_entries(
    df: pd.DataFrame,
    regex_substitutions: dict[str, str] | None = None,
    paragraph_newlines: int = 1,
    paragraph_token: str = PARAGRAPH_TOKEN
):
    """Canonicalize the text entries of a DataFrame."""
    df = df.copy()
    
    text_columns = [TITLE_COL, TEXT_COL] if TITLE_COL in df.columns else [TEXT_COL]
    
    for column in text_columns:
        df[column] = df[column].apply(canonicalize_text, regex_substitutions = regex_substitutions, paragraph_newlines = paragraph_newlines, paragraph_token = paragraph_token)
    
    return df

def drop_nonalphabetical_rows(df: pd.DataFrame):
    """Drop rows whose text does not contain letters."""
    LETTER_RE = re.compile(r"[^\W\d_]", flags = re.UNICODE)
    df = df.copy()
    
    def is_nonalphabetical(text):
        """Return whether the text does not contain letters."""
        return LETTER_RE.search(text) is None
    
    df.drop(df[df[TEXT_COL].apply(is_nonalphabetical)].index, inplace = True)
    
    return df

def drop_nonenglish_rows(df: pd.DataFrame):
    """Drop rows whose text or title is not in English."""
    df = df.copy()
    
    text_columns = [TITLE_COL, TEXT_COL] if TITLE_COL in df.columns else [TEXT_COL]
    
    def is_nonenglish(text):
        """Return whether the text is reliably detected to be non-English."""
        result = cld2.detect(str(text))
        
        if result[0] == False:
            return False
        
        return result[2][0][1] != "en"
    
    for column in text_columns:
        df.drop(df[df[column].apply(is_nonenglish)].index, inplace = True)
    
    return df

def drop_duplicate_rows(df: pd.DataFrame, match_title: bool = False):
    """Drop duplicate rows. If `match_title` is `True`, only additional rows with both matching text and title are dropped. Otherwise, all additional rows with just matching text are dropped.
    
    Args:
        df: The DataFrame to remove duplicate rows from.
        match_title: If `False`, all additional rows with matching text are removed.
    """
    df = df.copy()
    
    if match_title:
        df.drop_duplicates(subset = [TITLE_COL, TEXT_COL], inplace = True)
    else:
        df.drop_duplicates(subset = [TEXT_COL], inplace = True)
        
    return df

def reindex_rows(df: pd.DataFrame):
    """Reindex the rows of a DataFrame."""
    df = df.copy()
    
    df.index = range(len(df))
    
    return df

def clean_raw_dataframe(
    df: pd.DataFrame,
    regex_substitutions: dict[str, str] | None = None,
    paragraph_newlines: int = 1,
    paragraph_token: str = PARAGRAPH_TOKEN,
    match_title_for_duplicates: bool = False,
    drop_nonenglish: bool = True,
    reindex: bool = True
):
    """Apply the full cleaning pipeline to a raw DataFrame."""
    validate_raw_dataframe(df)
    df = df.copy()
    
    df = cast_text_columns_to_string(df)
    df = drop_invalid_label_rows(df)
    df = replace_whitespace_entries(df)
    df = drop_missing_text_rows(df)
    
    df = fill_na_with_empty_strings(df)
    df = canonicalize_text_entries(df, regex_substitutions = regex_substitutions, paragraph_newlines = paragraph_newlines, paragraph_token = paragraph_token)
    df = replace_whitespace_entries(df)
    df = drop_missing_text_rows(df)
    
    if drop_nonenglish:
        df = fill_na_with_empty_strings(df)
        df = drop_nonenglish_rows(df)
    
    df = drop_duplicate_rows(df, match_title_for_duplicates)
    
    if reindex:
        df = reindex_rows(df)
    
    df = fill_na_with_empty_strings(df)
    
    return df