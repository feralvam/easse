import pytest

from easse.fkgl import corpus_fkgl
from easse.cli.utils import read_file


def test_corpus_fkgl():
    assert corpus_fkgl(read_file("data/test_sets/turk/test.8turkers.tok.norm")) == pytest.approx(9.9, abs=1e-1)
    assert corpus_fkgl(read_file("data/test_sets/turk/test.8turkers.tok.turk.0")) == pytest.approx(8.2, abs=1e-1)
