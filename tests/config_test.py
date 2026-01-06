import pytest
import yaml

from fake_news_classifier.config import load_config

# load_config
def test_load_config_valid_yaml(tmp_path):
    config_data = {
        "model": {
            "name": "logistic_regression",
            "max_iter": 1000
        },
        "training": {
            "batch_size": 32
        }
    }

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config_data))

    result = load_config(str(config_file))

    assert result == config_data

def test_load_config_empty_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("")

    result = load_config(str(config_file))

    assert result is None

def test_load_config_invalid_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("model: [unclosed")

    with pytest.raises(yaml.YAMLError):
        load_config(str(config_file))

def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist.yaml")