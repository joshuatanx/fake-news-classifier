import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from fake_news_classifier.data.clean import (
    canonicalize_text_entries,
    cast_text_columns_to_string,
    clean_raw_dataframe,
    drop_duplicate_rows,
    drop_invalid_label_rows,
    drop_missing_text_rows,
    drop_nonalphabetical_rows,
    drop_nonenglish_rows,
    fill_na_with_empty_strings,
    fix_missing_spaces_around_punctuation,
    normalize_whitespace,
    reindex_rows,
    remove_ansi_codes,
    remove_control_codes,
    replace_whitespace_entries
)
from fake_news_classifier.data.schema import (
    LABEL_COL,
    TEXT_COL,
    TITLE_COL
)

# cast_text_columns_to_string
@pytest.mark.parametrize(
    "df, expected",
    [(
        pd.DataFrame({
            TITLE_COL: ["Title"],
            TEXT_COL: ["Text"],
            LABEL_COL: [0]
        }),
        pd.DataFrame({
            TITLE_COL: pd.Series(["Title"]),
            TEXT_COL: pd.Series(["Text"]),
            LABEL_COL: [0]
        })
    ), (
        pd.DataFrame({
            TITLE_COL: pd.Series([1], dtype = "int8"),
            TEXT_COL: pd.Series([True], dtype = "bool"),
            LABEL_COL: [0]
        }),
        pd.DataFrame({
            TITLE_COL: pd.Series(["1"]),
            TEXT_COL: pd.Series(["True"]),
            LABEL_COL: [0]
        })
    ), (
        pd.DataFrame({
            TITLE_COL: [1, "Title"],
            TEXT_COL: ["text", True],
            LABEL_COL: [0, 1]
        }),
        pd.DataFrame({
            TITLE_COL: pd.Series(["1", "Title"]),
            TEXT_COL: pd.Series(["text", "True"]),
            LABEL_COL: [0, 1]
        })
    ), (
        pd.DataFrame({
            TITLE_COL: [1, None],
            TEXT_COL: ["text", True],
            LABEL_COL: [0, 1]
        }),
        pd.DataFrame({
            TITLE_COL: pd.Series(["1", pd.NA]),
            TEXT_COL: pd.Series(["text", "True"]),
            LABEL_COL: [0, 1]
        })
    )]
)
def test_cast_text_columns_to_string(df, expected):
    assert_frame_equal(cast_text_columns_to_string(df), expected)

def test_cast_text_columns_to_string_does_not_mutate():
    df = pd.DataFrame({
        TITLE_COL: [1, None],
        TEXT_COL: ["text", True],
        LABEL_COL: [0, 1]
    })
    original = df.copy()
    
    cast_text_columns_to_string(df)
    
    assert_frame_equal(df, original)

# drop_invalid_label_rows
@pytest.mark.parametrize(
    "df, expected",
    [(
        pd.DataFrame({
            TEXT_COL: ["Text"],
            LABEL_COL: [0]
        }),
        pd.DataFrame({
            TEXT_COL: ["Text"],
            LABEL_COL: [0]
        })
    ), (
        pd.DataFrame({
            TEXT_COL: ["Text"],
            LABEL_COL: [1]
        }),
        pd.DataFrame({
            TEXT_COL: ["Text"],
            LABEL_COL: [1]
        })
    ), (
        pd.DataFrame({
            TEXT_COL: ["Text1", "Text2"],
            LABEL_COL: [2, 0]
        }),
        pd.DataFrame({
            TEXT_COL: ["Text2"],
            LABEL_COL: [0]
        }, index = [1])
    ), (
        pd.DataFrame({
            TEXT_COL: ["Text1", "Text2"],
            LABEL_COL: [-1, 0.5]
        }),
        pd.DataFrame({
            TEXT_COL: pd.Series(dtype = "object"),
            LABEL_COL: pd.Series(dtype = "float64")
        })
    )]
)
def test_drop_invalid_label_rows(df, expected):
    assert_frame_equal(drop_invalid_label_rows(df), expected)

def test_drop_invalid_label_rows_does_not_mutate():
    df = pd.DataFrame({
        TEXT_COL: ["Text", "Text"],
        LABEL_COL: [0, 2]
    })
    original = df.copy()
    
    drop_invalid_label_rows(df)
    
    assert_frame_equal(df, original)

# replace_whitespace_entries
@pytest.mark.parametrize(
    "df, expected",
    [(
        pd.DataFrame({
            TITLE_COL: ["Title1"],
            TEXT_COL: ["Text1"],
            LABEL_COL: [0]
        }),
        pd.DataFrame({
            TITLE_COL: ["Title1"],
            TEXT_COL: ["Text1"],
            LABEL_COL: [0]
        })
    ), (
        pd.DataFrame({
            TITLE_COL: ["Title1"],
            TEXT_COL: [" "],
            LABEL_COL: [0]
        }),
        pd.DataFrame({
            TITLE_COL: ["Title1"],
            TEXT_COL: [pd.NA],
            LABEL_COL: [0]
        })
    ), (
        pd.DataFrame({
            TITLE_COL: [" "],
            TEXT_COL: ["Text1"],
            LABEL_COL: [0]
        }),
        pd.DataFrame({
            TITLE_COL: [pd.NA],
            TEXT_COL: ["Text1"],
            LABEL_COL: [0]
        })
    )]
)
def test_replace_whitespace_entries(df, expected):
    assert_frame_equal(replace_whitespace_entries(df), expected)

def test_replace_whitespace_entries_does_not_mutate():
    df = pd.DataFrame({
        TITLE_COL: [" "],
        TEXT_COL: ["Text1"],
        LABEL_COL: [0]
    })
    original = df.copy()
    
    replace_whitespace_entries(df)
    
    assert_frame_equal(df, original)

# drop_missing_text_rows
@pytest.mark.parametrize(
    "df, expected",
    [(
        pd.DataFrame({
            TITLE_COL: ["Title1", "Title2"],
            TEXT_COL: ["Text1", "Text2"],
            LABEL_COL: [0, 0]
        }),
        pd.DataFrame({
            TITLE_COL: ["Title1", "Title2"],
            TEXT_COL: ["Text1", "Text2"],
            LABEL_COL: [0, 0]
        })
    ), (
        pd.DataFrame({
            TITLE_COL: ["Title1", None],
            TEXT_COL: ["Text1", "Text2"],
            LABEL_COL: [0, 0]
        }),
        pd.DataFrame({
            TITLE_COL: ["Title1", None],
            TEXT_COL: ["Text1", "Text2"],
            LABEL_COL: [0, 0]
        })
    ), (
        pd.DataFrame({
            TITLE_COL: ["Title1", "Title2"],
            TEXT_COL: ["Text1", None],
            LABEL_COL: [0, 0]
        }),
        pd.DataFrame({
            TITLE_COL: ["Title1"],
            TEXT_COL: ["Text1"],
            LABEL_COL: [0]
        })
    )]
)
def test_drop_missing_text_rows(df, expected):
    assert_frame_equal(drop_missing_text_rows(df), expected)

def test_drop_missing_text_rows_does_not_mutate():
    df = pd.DataFrame({
        TITLE_COL: ["Title1", "Title2"],
        TEXT_COL: ["Text1", None],
        LABEL_COL: [0, 0]
    })
    original = df.copy()
    
    drop_missing_text_rows(df)
    
    assert_frame_equal(df, original)

# fill_na_with_empty_strings
@pytest.mark.parametrize(
    "df, expected",
    [(
        pd.DataFrame({
            TITLE_COL: ["Title"],
            TEXT_COL: ["Text"],
            LABEL_COL: [0]
        }),
        pd.DataFrame({
            TITLE_COL: ["Title"],
            TEXT_COL: ["Text"],
            LABEL_COL: [0]
        })
    ), (
        pd.DataFrame({
            TITLE_COL: [pd.NA],
            TEXT_COL: [None],
            LABEL_COL: [0]
        }),
        pd.DataFrame({
            TITLE_COL: [""],
            TEXT_COL: [""],
            LABEL_COL: [0]
        })
    )]
)
def test_fill_na_with_empty_strings(df, expected):
    assert_frame_equal(fill_na_with_empty_strings(df), expected)

def test_fill_na_with_empty_strings_does_not_mutate():
    df = pd.DataFrame({
        TITLE_COL: [pd.NA],
        TEXT_COL: [None],
        LABEL_COL: [0]
    })
    original = df.copy()
    
    fill_na_with_empty_strings(df)
    
    assert_frame_equal(df, original)

# remove_ansi_codes
@pytest.mark.parametrize(
    "text, expected",
    [(
        "\x1b[31mRed text\x1b[0m",
        " Red text "
    ), (
        "Error:\x1b[31m failed\x1b[0m at step 3",
        "Error:  failed  at step 3"
    ), (
        "\x1b[2J\x1b[HScreen cleared",
        "  Screen cleared"
    ), (
        "\x1b7Saved\x1b8Restored",
        " Saved Restored"
    ), (
        "[INFO] \x1b[32mSuccess\x1b[0m",
        "[INFO]  Success "
    )]
)
def test_remove_ansi_codes(text, expected):
    assert remove_ansi_codes(text) == expected

# remove_control_codes
@pytest.mark.parametrize(
    "text, expected",
    [
    (
        "Spacing\tCodes\nNext\rLine",
        "Spacing Codes Next Line"
    ), (
        "Space\vB\fC",
        "Space B C"
    ), (
        "C0\x00\x01\x02Controls",
        "C0Controls"
    ), (
        "DEL\x7fWorld",
        "DELWorld"
    ), (
        "C1\x85\x90Controls",
        "C1Controls"
    ), (
        "\ufeffBOM",
        "BOM"
    ), (
        "zero\u200bwidth\u200ctext\u200d",
        "zerowidthtext"
    ), (
        "A\tB\x00C\nD\x7fE\u200bF",
        "A BC DEF"
    )]
)
def test_remove_control_codes(text, expected):
    assert remove_control_codes(text) == expected

# fix_missing_spaces_around_punctuation
@pytest.mark.parametrize(
    "text, expected",
    [(
        # Split after acronym chains
        "The U.S.President spoke today.",
        "The U.S. President spoke today.",
    ), (
        "I met the U.K.Prime Minister.",
        "I met the U.K. Prime Minister.",
    ), (   # Missing space after closing brackets/parenthesis/brace
        "(Reuters)WASHINGTON - Report.",
        "(Reuters) WASHINGTON - Report.",
    ), (
        "[AP]NEW YORK is big.",
        "[AP] NEW YORK is big.",
    ), (
        "{Note}Hello there.",
        "{Note} Hello there.",
    ), (
        # Missing space after sentence punctuation when next starts with a capital or opening bracket
        "Hello.World",
        "Hello. World",
    ), (
        "Wait!What happened?",
        "Wait! What happened?",
    ), (
        "Is this real?Yes it is.",
        "Is this real? Yes it is.",
    ), (
        "Done.(More to come)",
        "Done. (More to come)",
    ), (
        "Done[More to come]",
        "Done [More to come]",
    ), (
        "Done{More to come}",
        "Done {More to come}",
    ), (
        # Protect URLs
        "See https://example.com/Hello.World for more.",
        "See https://example.com/Hello.World for more.",
    ), (
        "Visit www.example.com/Hello.WorldNow",
        "Visit www.example.com/Hello.WorldNow",
    ), (
        # Protect emails
        "Email john.doe@example.comNow please.",
        "Email john.doe@example.comNow please.",
    ),
    (
        "Contact support@example.co.ukASAP!",
        "Contact support@example.co.ukASAP!",
    ), (
        # Mixed case
        "(Reuters)WASHINGTONHello.World U.S.President met (AP)NEW YORK.",
        "(Reuters) WASHINGTONHello. World U.S. President met (AP) NEW YORK.",
    )]
)
def test_fix_missing_spaces_around_punctuation(text, expected):
    assert fix_missing_spaces_around_punctuation(text) == expected

# normalize_whitespace
@pytest.mark.parametrize(
    "text, expected",
    [
        ("  ", ""),
        ("Call  me", "Call me"),
        (" Hello world ", "Hello world"),
        ("Hello world", "Hello world")
    ]
)
def test_normalize_whitespace(text, expected):
    assert normalize_whitespace(text) == expected

# canonicalize_text_entries
@pytest.mark.parametrize(
    "df, expected",
    [(
        pd.DataFrame({
            TITLE_COL: ["\x1b[31mRed\tTitle\x1b[0m", "(Reuters)WASHINGTON"],
            TEXT_COL: ["Hello\x00World\nNext", "The U.S.President said:Hi!What?"]
        }),
        pd.DataFrame({
            TITLE_COL: ["Red Title", "(Reuters) WASHINGTON"],
            TEXT_COL: ["HelloWorld Next", "The U.S. President said:Hi! What?"]
        })
    )]
)
def test_canonicalize_text_entries(df, expected):
    assert_frame_equal(canonicalize_text_entries(df), expected)

def test_canonicalize_text_entries_does_not_mutate():
    df = pd.DataFrame({
        TITLE_COL: ["\x1b[31mRed\tTitle\x1b[0m", "(Reuters)WASHINGTON"],
        TEXT_COL: ["Hello\x00World\nNext", "The U.S.President said:Hi!What?"]
    })
    original = df.copy()
    
    canonicalize_text_entries(df)
    
    assert_frame_equal(df, original)

# drop_nonalphabetical_rows
@pytest.mark.parametrize(
    "df, expected",
    [(
        pd.DataFrame({
            TITLE_COL: ["Hello word"],
            TEXT_COL: ["Hello world"],
            LABEL_COL: [0]
        }), pd.DataFrame({
            TITLE_COL: ["Hello word"],
            TEXT_COL: ["Hello world"],
            LABEL_COL: [0]
        })
    ), (
        pd.DataFrame({
            TITLE_COL: ["Title", "1"],
            TEXT_COL: ["Text", "Hello world"],
            LABEL_COL: [0, 0]
        }), pd.DataFrame({
            TITLE_COL: ["Title", "1"],
            TEXT_COL: ["Text", "Hello world"],
            LABEL_COL: [0, 0]
        })
    ), (
        pd.DataFrame({
            TITLE_COL: ["Hello world", "Title"],
            TEXT_COL: ["Text", "20"],
            LABEL_COL: [0, 0]
        }), pd.DataFrame({
            TITLE_COL: ["Hello world"],
            TEXT_COL: ["Text"],
            LABEL_COL: [0]
        })
    )]
)
def test_drop_nonalphabetical_rows(df, expected):
    assert_frame_equal(drop_nonalphabetical_rows(df), expected)

def test_drop_nonalphabetical_rows_does_not_mutate():
    df = pd.DataFrame({
        TITLE_COL: ["Hello world", "Title"],
        TEXT_COL: ["Text", "20"],
        LABEL_COL: [0, 0]
    })
    original = df.copy()
    
    drop_nonalphabetical_rows(df)
    
    assert_frame_equal(df, original)

# drop_nonenglish_rows
@pytest.mark.parametrize(
    "df, expected",
    [(
        # Non-English in text and title
        pd.DataFrame({
            TEXT_COL: ["This is an English sentence.", "Ceci est une phrase française."],
            TITLE_COL: ["English title", "Titre français"],
            LABEL_COL: [0, 1]
        }),
        pd.DataFrame({
            TEXT_COL: ["This is an English sentence."],
            TITLE_COL: ["English title"],
            LABEL_COL: [0]
        })
    ), (
        # Non-English text
        pd.DataFrame({
            TEXT_COL: ["This is English.", "Esto es español."],
            LABEL_COL: [0, 1]
        }),
        pd.DataFrame({
            TEXT_COL: ["This is English."],
            LABEL_COL: [0]
        })
    ), (
        # All English
        pd.DataFrame({
            TEXT_COL: ["Breaking news today.", "Another English sentence."],
            TITLE_COL: ["News update", "More news"],
            LABEL_COL: [0, 1]
        }),
        pd.DataFrame({
            TEXT_COL: ["Breaking news today.", "Another English sentence."],
            TITLE_COL: ["News update", "More news"],
            LABEL_COL: [0, 1]
        })
    )]
)
def test_drop_nonenglish_rows(df, expected):
    assert_frame_equal(drop_nonenglish_rows(df), expected)

def test_drop_nonenglish_rows_does_not_mutate():
    df = pd.DataFrame({
        TEXT_COL: ["This is English.", "Esto es español."],
        LABEL_COL: [0, 1]
    })
    original = df.copy()
    
    drop_nonenglish_rows(df)
    
    assert_frame_equal(df, original)

# drop_duplicate_rows
@pytest.mark.parametrize(
    "df, expected",
    [(
        pd.DataFrame({
            TITLE_COL: ["Title1", "Title2"],
            TEXT_COL: ["Text1", "Text2"],
            LABEL_COL: [0, 0]
        }),
        pd.DataFrame({
            TITLE_COL: ["Title1", "Title2"],
            TEXT_COL: ["Text1", "Text2"],
            LABEL_COL: [0, 0]
        })
    ), (
        pd.DataFrame({
            TITLE_COL: ["Title", "Title"],
            TEXT_COL: ["Text1", "Text2"],
            LABEL_COL: [0, 0]
        }),
        pd.DataFrame({
            TITLE_COL: ["Title", "Title"],
            TEXT_COL: ["Text1", "Text2"],
            LABEL_COL: [0, 0]
        })
    ), (
        pd.DataFrame({
            TITLE_COL: ["Title1", "Title2"],
            TEXT_COL: ["Text", "Text"],
            LABEL_COL: [0, 0]
        }),
        pd.DataFrame({
            TITLE_COL: ["Title1"],
            TEXT_COL: ["Text"],
            LABEL_COL: [0]
        })
    ), (
        pd.DataFrame({
            TITLE_COL: ["Title", "Title"],
            TEXT_COL: ["Text", "Text"],
            LABEL_COL: [0, 0]
        }),
        pd.DataFrame({
            TITLE_COL: ["Title"],
            TEXT_COL: ["Text"],
            LABEL_COL: [0]
        })
    )]
)
def test_drop_duplicate_rows_match_text(df, expected):
    assert_frame_equal(drop_duplicate_rows(df), expected)

@pytest.mark.parametrize(
    "df, expected",
    [(
        pd.DataFrame({
            TITLE_COL: ["Title1", "Title2"],
            TEXT_COL: ["Text1", "Text2"],
            LABEL_COL: [0, 0]
        }),
        pd.DataFrame({
            TITLE_COL: ["Title1", "Title2"],
            TEXT_COL: ["Text1", "Text2"],
            LABEL_COL: [0, 0]
        })
    ), (
        pd.DataFrame({
            TITLE_COL: ["Title", "Title"],
            TEXT_COL: ["Text1", "Text2"],
            LABEL_COL: [0, 0]
        }),
        pd.DataFrame({
            TITLE_COL: ["Title", "Title"],
            TEXT_COL: ["Text1", "Text2"],
            LABEL_COL: [0, 0]
        })
    ), (
        pd.DataFrame({
            TITLE_COL: ["Title1", "Title2"],
            TEXT_COL: ["Text", "Text"],
            LABEL_COL: [0, 0]
        }),
        pd.DataFrame({
            TITLE_COL: ["Title1", "Title2"],
            TEXT_COL: ["Text", "Text"],
            LABEL_COL: [0, 0]
        })
    ), (
        pd.DataFrame({
            TITLE_COL: ["Title", "Title"],
            TEXT_COL: ["Text", "Text"],
            LABEL_COL: [0, 0]
        }),
        pd.DataFrame({
            TITLE_COL: ["Title"],
            TEXT_COL: ["Text"],
            LABEL_COL: [0]
        })
    )]
)
def test_drop_duplicate_rows_match_title(df, expected):
    assert_frame_equal(drop_duplicate_rows(df, True), expected)

def test_drop_duplicate_rows_does_not_mutate():
    df = pd.DataFrame({
        TITLE_COL: ["Title1", "Title2"],
        TEXT_COL: ["Text", "Text"],
        LABEL_COL: [0, 0]
    })
    original = df.copy()
    
    drop_duplicate_rows(df)
    
    assert_frame_equal(df, original)

# reindex_rows
@pytest.mark.parametrize(
    "df, expected",
    [(
        # Non-default indexing
        pd.DataFrame({
            TEXT_COL: ["text", "text"],
            LABEL_COL: [0, 1]
        }, index = [2, 4]),
        pd.DataFrame({
            TEXT_COL: ["text", "text"],
            LABEL_COL: [0, 1]
        })
    ), (
        # Default indexing
        pd.DataFrame({
            TEXT_COL: ["text", "text"],
            LABEL_COL: [0, 1]
        }),
        pd.DataFrame({
            TEXT_COL: ["text", "text"],
            LABEL_COL: [0, 1]
        })
    )]
)
def test_reindex_rows(df, expected):
    assert_frame_equal(reindex_rows(df), expected)

def test_reindex_rows_does_not_mutate():
    df = pd.DataFrame({
        TEXT_COL: ["text", "text"],
        LABEL_COL: [0, 1]
    }, index = [2, 4])
    original = df.copy()
    
    reindex_rows(df)
    
    assert_frame_equal(df, original)

# clean_raw_dataframe
@pytest.mark.parametrize(
    "match_title_for_duplicates, expected",
    [(
        False,
        pd.DataFrame({
            LABEL_COL: [1],
            TITLE_COL: ["Title A"],
            TEXT_COL: ["Hi there"]
        }),
    ),
    (
        True,
        pd.DataFrame({
            LABEL_COL: [1, 1],
            TITLE_COL: ["Title A", "Title B"],
            TEXT_COL: ["Hi there", "Hi there"],
        }),
    )]
)
def test_clean_raw_dataframe(match_title_for_duplicates, expected):
    df = pd.DataFrame({
        LABEL_COL: [2, 0, 1, 1],
        TITLE_COL: ["Bad", "Whitespace", "Title A", "Title B"],
        TEXT_COL: [
            "Valid text but bad label",
            "   \t  \n",                    # whitespace-only -> NA -> drop
            "\x1b[31mHi   there\x1b[0m",    # ANSI + repeated whitespace -> "Hi there"
            "Hi there",                     # potential duplicate
        ]
    }, index = [10, 11, 12, 13])

    expected = expected.copy()
    expected.index = range(len(expected))
    assert_frame_equal(clean_raw_dataframe(df, match_title_for_duplicates = match_title_for_duplicates), expected)

def test_clean_raw_dataframe_does_not_mutate_input():
    df = pd.DataFrame({
        LABEL_COL: [0, 1],
        TITLE_COL: ["T1", "T2"],
        TEXT_COL: ["Hello", "World"]
    }, index = [5, 9])
    original = df.copy(deep = True)

    clean_raw_dataframe(df)

    assert_frame_equal(df, original)