import pandas as pd
import pytest

from fake_news.data.schema import (
    CONTENT_COL,
    LABEL_COL,
    TEXT_COL,
    TITLE_COL,
    validate_dataframe,
    validate_raw_dataframe
)

# validate_raw_dataframe
@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({
            TITLE_COL: ["title"],
            TEXT_COL: ["text"],
            LABEL_COL: [0],
            CONTENT_COL: ["content"]
        }),
        pd.DataFrame({
            TEXT_COL: [],
            LABEL_COL: []
        }),
        pd.DataFrame({
            TITLE_COL: ["title"],
            TEXT_COL: ["text"],
            LABEL_COL: [1],
            "column": ["hello"]
        }),
        pd.DataFrame({
            TITLE_COL: [1],
            TEXT_COL: ["text"],
            LABEL_COL: [1]
        }),
    ]
)
def test_validate_raw_dataframe_accepts_valid(df):
    validate_raw_dataframe(df)

@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({
            TITLE_COL: ["title"],
            LABEL_COL: [0]
        }),
        pd.DataFrame({
            TEXT_COL: ["text"]
        })
    ]
)
def test_validate_raw_dataframe_raises_valueerror(df):
    with pytest.raises(ValueError):
        validate_raw_dataframe(df)

@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({
            TITLE_COL: [["title", "title"]],
            TEXT_COL: ["text"],
            LABEL_COL: [0]
        }),
        pd.DataFrame({
            TITLE_COL: ["title"],
            TEXT_COL: [{"key1": "text1", "key2": "text2"}],
            LABEL_COL: [0]
        }),
    ]
)
def test_validate_raw_dataframe_raises_typeerror(df):
    with pytest.raises(TypeError):
        validate_raw_dataframe(df)

# validate_dataframe
@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({
            TITLE_COL: ["title"],
            TEXT_COL: ["text"],
            LABEL_COL: [0]
        }),
        pd.DataFrame({
            TITLE_COL: ["title1", "title2"],
            TEXT_COL: ["text1", "text2"],
            LABEL_COL: [0, 1]
        }),
        pd.DataFrame({
            TEXT_COL: ["text1", "text2"],
            LABEL_COL: [0, 0]
        }),
        pd.DataFrame({
            TITLE_COL: ["title1", "title2"],
            TEXT_COL: ["text1", "text2"],
            LABEL_COL: [0, 1],
            CONTENT_COL: ["content1", "content2"]
        }),
        pd.DataFrame({
            TITLE_COL: [],
            TEXT_COL: [],
            LABEL_COL: []
        }),
    ]
)
def test_validate_dataframe_accepts_valid(df):
    validate_dataframe(df)

@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({
            TITLE_COL: ["title"],
            LABEL_COL: [0]
        }),
        pd.DataFrame({
            TEXT_COL: ["text"]
        }),
        pd.DataFrame({
            TITLE_COL: ["title"],
            TEXT_COL: [None],
            LABEL_COL: [0]
        }),
        pd.DataFrame({
            TITLE_COL: [None],
            TEXT_COL: [None],
            LABEL_COL: [None]
        }),
        pd.DataFrame({
            TITLE_COL: ["title"],
            TEXT_COL: ["text"],
            LABEL_COL: [2]
        })
    ]
)
def test_validate_dataframe_raises_valueerror(df):
    with pytest.raises(ValueError):
        validate_dataframe(df)
        
@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({
            TITLE_COL: [["title", "title"]],
            TEXT_COL: ["text"],
            LABEL_COL: [0]
        }),
        pd.DataFrame({
            TITLE_COL: ["title"],
            TEXT_COL: [{"key1": "text1", "key2": "text2"}],
            LABEL_COL: [0]
        }),
    ]
)
def test_validate_dataframe_raises_typeerror(df):
    with pytest.raises(TypeError):
        validate_dataframe(df)