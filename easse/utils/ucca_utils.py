from functools import lru_cache
from typing import List

from tupa.parse import Parser
from ucca.core import Passage
import ucca.convert

from easse.utils.constants import UCCA_PARSER_PATH
from easse.utils.resources import download_ucca_model


@lru_cache(maxsize=1)
def get_parser():
    if not UCCA_PARSER_PATH.parent.exists():
        download_ucca_model()
    return Parser(str(UCCA_PARSER_PATH))


def ucca_parse_texts(texts: List[str]):
    passages = list(ucca.convert.from_text(texts, one_per_line=True))
    parser = get_parser()
    parsed_passages = [passage for (passage, *_) in parser.parse(passages, display=False)]
    return parsed_passages


def get_scenes(ucca_passage: Passage):
    """Return all the ucca scenes in the given text"""
    ucca_scenes = [x for x in ucca_passage.layer('1').all if x.tag == "FN" and x.is_scene()]
    text_scenes = []
    for scene in ucca_scenes:
        words = []
        previous_word = ''
        for terminal in scene.get_terminals(False, True):
            word = terminal.text
            if word == previous_word:
                # TODO: Iterating this way on the scene sometimes yields duplicates.
                continue
            words.append(word)
            previous_word = word
        text_scenes.append(words)
    return text_scenes
