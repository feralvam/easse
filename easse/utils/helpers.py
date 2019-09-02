from pathlib import Path
import tempfile


def get_temp_filepath(create=False):
    temp_filepath = Path(tempfile.mkstemp()[1])
    if not create:
        temp_filepath.unlink()
    return temp_filepath


def read_lines(filename):
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


def add_dicts(*dicts):
    return {k: v for dic in dicts for k, v in dic.items()}


def safe_division(a, b):
    if b == 0:
        return 0
    return a / b
