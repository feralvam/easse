import pytest

from easse.fkgl import corpus_fkgl
from easse.utils.resources import get_orig_sents, get_refs_sents


def test_corpus_fkgl():
    assert corpus_fkgl(get_orig_sents('turkcorpus_test_legacy')) == pytest.approx(9.9, abs=1e-1)
    assert corpus_fkgl(get_refs_sents('turkcorpus_test_legacy')[0]) == pytest.approx(8.2, abs=1e-1)
