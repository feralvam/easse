import pytest

from easse.bleu import corpus_averaged_sentence_bleu
from easse.utils.helpers import read_split_lines, collapse_split_sentences
from easse.utils.constants import DATA_DIR


def test_compute_macro_avg_sent_bleu():
    sys_sents = read_split_lines(DATA_DIR / "test_sets/wikisplit/wikisplit.test.untok.split")
    refs_sents = [read_split_lines(DATA_DIR / "test_sets/wikisplit/wikisplit.test.untok.split")]

    collapsed_sys_sents, collapsed_refs_sents = collapse_split_sentences(sys_sents, refs_sents)

    sent_bleu = corpus_averaged_sentence_bleu(collapsed_sys_sents, collapsed_refs_sents)
    assert sent_bleu == pytest.approx(100.0)
