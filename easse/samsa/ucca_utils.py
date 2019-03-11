from tupa.parse import Parser
import ucca.convert

PARSER_PATH = "resources/ucca/models/ucca-bilstm"
PARSER = None


def get_parser():
    global PARSER
    if PARSER is None:
        PARSER = Parser(PARSER_PATH)
    return PARSER


def ucca_parse_text(text):
    text = ucca.convert.from_text(text, one_per_line=True)
    text = list(text)
    parser = get_parser()
    ucca_passages = [passage for (passage, *_) in parser.parse(text, display=False)]
    return ucca_passages[0]


def get_scenes(ucca_passage):
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
