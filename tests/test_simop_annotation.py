from easse.annotation.word_level import WordOperationAnnotator
from easse.annotation.sentence_level import SentenceOperationAnnotator
from easse.utils.helpers import read_lines
from easse.utils.resources import (
    get_orig_sents,
    get_refs_sents,
    get_system_outputs_dir,
)


@pytest.mark.skip(reason="TODO: Add assert")
def test_identification_wordlevel():
    orig_sent = "The tarantula, the trickster character, spun a black cord and, attaching it to the ball, crawled away fast to the east, pulling on the cord with all his strength."
    simp_sent = "The tarantula was a trickster character. He spun a black cord. He attached it to the ball. He then crawled away fast to the east. As he crawled, he was pulling on the cord with all his strength."

    word_simop_annot = WordOperationAnnotator(tokenizer="moses", lowercase=False)
    orig_labels, simp_labels = word_simop_annot.identify_operations([orig_sent], [simp_sent])
    print(orig_labels)
    print(simp_labels)


@pytest.mark.skip(reason="TODO: Add assert")
def test_identification_sentencelevel():
    sent_simop_annot = SentenceOperationAnnotator(tokenizer="moses", lowercase=False)
    orig_sent = "In return, Rollo swore fealty to Charles, converted to Christianity, and undertook to defend the northern region of France against the incursions of other Viking groups."
    simp_sent = "In return, Rollo swore fealty to Charles, converted to Christianity, and set out to defend the north of France from the raids of other Viking groups."
    labels = sent_simop_annot.identify_operations([orig_sent], [simp_sent])
    print(labels)


@pytest.mark.skip(reason="TODO: Add assert")
def test_analysis_score():
    orig_sents = [
        "About 95 species are currently accepted.",
        "The cat perched on the mat.",
    ]
    sys_sents = ["About 95 you now get in.", "Cat on mat."]
    refs_sents = [
        ["About 95 species are currently known.", "The cat sat on the mat."],
        ["About 95 species are now accepted.", "The cat is on the mat."],
        ["95 species are now accepted.", "The cat sat."],
    ]

    word_simop_annot = WordOperationAnnotator(tokenizer="moses", lowercase=False, verbose=True)
    scores = word_simop_annot.analyse_operations(orig_sents, sys_sents, refs_sents)
    print(scores)
