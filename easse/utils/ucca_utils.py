from contextlib import contextmanager
from functools import lru_cache
import sys
from typing import List

from tupa.parse import Parser
from ucca.core import Passage
import ucca.convert

from easse.utils.constants import UCCA_PARSER_PATH
from easse.utils.resources import download_ucca_model


@contextmanager
def mock_sys_argv(argv):
    original_sys_argv = sys.argv
    sys.argv = argv
    yield
    sys.argv = original_sys_argv


@lru_cache(maxsize=1)
def get_parser():
    if not UCCA_PARSER_PATH.parent.exists():
        download_ucca_model()
    with mock_sys_argv(['']):
        # Need to mock sysargs otherwise the parser will use try to use them and throw an exception
        return Parser(str(UCCA_PARSER_PATH))


def ucca_parse_texts(texts: List[str]):
    passages = list(ucca.convert.from_text(texts, one_per_line=True))
    parser = get_parser()
    parsed_passages = [passage for (passage, *_) in parser.parse(passages, display=False)]
    return parsed_passages


def get_scenes_ucca(ucca_passage: Passage):
    return [x for x in ucca_passage.layer('1').all if x.tag == "FN" and x.is_scene()]


def get_scenes_text(ucca_passage: Passage):
    """Return all the ucca scenes in the given text"""
    scenes_ucca = get_scenes_ucca(ucca_passage)
    scenes_text = []
    for scene in scenes_ucca:
        words = []
        previous_word = ''
        for terminal in scene.get_terminals(False, True):
            word = terminal.text
            if word == previous_word:
                # TODO: Iterating this way on the scene sometimes yields duplicates.
                continue
            words.append(word)
            previous_word = word
        scenes_text.append(words)
    return scenes_text
