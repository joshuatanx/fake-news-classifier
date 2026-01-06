import yaml

def load_config(path: str):
    """Return the parsed data of a YAML config file."""
    with open(path, "r") as stream:
        data = yaml.safe_load(stream)
    
    return data