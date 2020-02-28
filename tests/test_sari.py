import pytest

from easse import sari
from easse.utils.helpers import read_lines
from easse.utils.constants import DATA_DIR
from easse.utils.resources import get_orig_sents, get_refs_sents, get_system_outputs_dir


@pytest.mark.skip(reason='TODO: Test probably broken, need to investigate')
def test_sari_sentence():
    orig_sent = "About 95 species are currently accepted ."
    refs_sents = ["About 95 species are currently known .",
                 "About 95 species are now accepted .",
                 "95 species are now accepted ."]

    sys_sent = "About 95 you now get in ."
    sari_score = sari.sari_sentence(orig_sent, sys_sent, refs_sents)
    assert sari_score == pytest.approx(26.82782411698074)

    sys_sent = "About 95 species are now agreed ."
    sari_score = sari.sari_sentence(orig_sent, sys_sent, refs_sents)
    assert sari_score == pytest.approx(58.89995423074248)

    sys_sent = "About 95 species are currently agreed ."
    sari_score = sari.sari_sentence(orig_sent, sys_sent, refs_sents)
    assert sari_score == pytest.approx(50.71608864657479)


def test_corpus_sari_plain():
    orig_sents = get_orig_sents('turkcorpus_test')
    refs_sents = get_refs_sents('turkcorpus_test')

    system_outputs_dir = get_system_outputs_dir('turkcorpus_test')
    hyp_sents = read_lines(system_outputs_dir / "lower/Dress-Ls.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, refs_sents, tokenizer='plain')
    assert sari_score == pytest.approx(36.73586275692667)

    hyp_sents = read_lines(system_outputs_dir / "lower/Dress.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, refs_sents, tokenizer='plain')
    assert sari_score == pytest.approx(36.5859900146575)

    hyp_sents = read_lines(system_outputs_dir / "lower/EncDecA.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, refs_sents, tokenizer='plain')
    assert sari_score == pytest.approx(34.73946658449856)

    hyp_sents = read_lines(system_outputs_dir / "lower/Hybrid.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, refs_sents, tokenizer='plain')
    assert sari_score == pytest.approx(31.008109926854227)

    hyp_sents = read_lines(system_outputs_dir / "lower/PBMT-R.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, refs_sents, tokenizer='plain')
    assert sari_score == pytest.approx(37.817966679481013)

    hyp_sents = read_lines(system_outputs_dir / "lower/SBMT-SARI.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, refs_sents, tokenizer='plain')
    assert sari_score == pytest.approx(39.360477024519125)


def test_corpus_sari_tokenize():
    orig_sents = get_orig_sents('turkcorpus_test')
    refs_sents = get_refs_sents('turkcorpus_test')
    system_outputs_dir = get_system_outputs_dir('turkcorpus_test')

    hyp_sents = read_lines(system_outputs_dir / "lower/Dress-Ls.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, refs_sents)
    assert sari_score == pytest.approx(37.266058818588216)

    hyp_sents = read_lines(system_outputs_dir / "lower/Dress.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, refs_sents)
    assert sari_score == pytest.approx(37.08210095744638)

    hyp_sents = read_lines(system_outputs_dir / "lower/EncDecA.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, refs_sents)
    assert sari_score == pytest.approx(35.65754396121206)

    hyp_sents = read_lines(system_outputs_dir / "lower/Hybrid.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, refs_sents)
    assert sari_score == pytest.approx(31.39665078989411)

    hyp_sents = read_lines(system_outputs_dir / "lower/PBMT-R.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, refs_sents)
    assert sari_score == pytest.approx(38.558843050332037)

    hyp_sents = read_lines(system_outputs_dir / "lower/SBMT-SARI.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, refs_sents)
    assert sari_score == pytest.approx(39.964857928109127)
