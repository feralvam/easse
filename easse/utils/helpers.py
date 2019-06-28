import json
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
