from typing import List
from pathlib import Path
import tempfile

def safe_divide(a,b):
    return a/b if b else 0

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


def read_split_lines(filename, split_sep='<::::>'):
    with open(filename, encoding="utf-8") as f:
        split_instances = []
        for line in f:
            split_instances.append([split.strip() for split in line.split(split_sep)])

    return split_instances


def collapse_split_sentences(sys_sents: List[List[str]], refs_sents: List[List[List[str]]]):
    collapsed_sys_splits = [' '.join(sys_splits) for sys_splits in sys_sents]
    collapsed_refs_splits = [[' '.join(ref_splits) for ref_splits in ref_sents] for ref_sents in refs_sents]

    return collapsed_sys_splits, collapsed_refs_splits
