import pytest

from easse import sari
from easse.utils.helpers import read_lines


# def test_sari_sentence():
#     orig_sent = "About 95 species are currently accepted ."
#     ref_sents = ["About 95 species are currently known .",
#                  "About 95 species are now accepted .",
#                  "95 species are now accepted ."]
#
#     sys_sent = "About 95 you now get in ."
#     sari_score = sari.sari_sentence(orig_sent, sys_sent, ref_sents)
#     assert sari_score == pytest.approx(26.82782411698074)
#
#     sys_sent = "About 95 species are now agreed ."
#     sari_score = sari.sari_sentence(orig_sent, sys_sent, ref_sents)
#     assert sari_score == pytest.approx(58.89995423074248)
#
#     sys_sent = "About 95 species are currently agreed ."
#     sari_score = sari.sari_sentence(orig_sent, sys_sent, ref_sents)
#     assert sari_score == pytest.approx(50.71608864657479)


def test_corpus_sari_plain():
    orig_sents = read_lines("data/test_sets/turk/test.8turkers.tok.norm")
    ref_sents = []
    for n in range(8):
        ref_lines = read_lines(f"data/test_sets/turk/test.8turkers.tok.turk.{n}")
        ref_sents.append(ref_lines)

    hyp_sents = read_lines("data/system_outputs/turk/lower/Dress-Ls.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, ref_sents, tokenizer='plain')
    assert sari_score == pytest.approx(36.73586275692667)

    hyp_sents = read_lines("data/system_outputs/turk/lower/Dress.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, ref_sents, tokenizer='plain')
    assert sari_score == pytest.approx(36.5859900146575)

    hyp_sents = read_lines("data/system_outputs/turk/lower/EncDecA.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, ref_sents, tokenizer='plain')
    assert sari_score == pytest.approx(34.73946658449856)

    hyp_sents = read_lines("data/system_outputs/turk/lower/Hybrid.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, ref_sents, tokenizer='plain')
    assert sari_score == pytest.approx(31.008109926854227)

    hyp_sents = read_lines("data/system_outputs/turk/lower/PBMT-R.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, ref_sents, tokenizer='plain')
    assert sari_score == pytest.approx(37.817966679481013)

    hyp_sents = read_lines("data/system_outputs/turk/lower/SBMT-SARI.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, ref_sents, tokenizer='plain')
    assert sari_score == pytest.approx(39.360477024519125)


def test_corpus_sari_tokenize():
    orig_sents = read_lines("data/test_sets/turk/test.8turkers.tok.norm")
    ref_sents = []
    for n in range(8):
        ref_lines = read_lines(f"data/test_sets/turk/test.8turkers.tok.turk.{n}")
        ref_sents.append(ref_lines)

    hyp_sents = read_lines("data/system_outputs/turk/lower/Dress-Ls.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, ref_sents)
    assert sari_score == pytest.approx(37.266058818588216)

    hyp_sents = read_lines("data/system_outputs/turk/lower/Dress.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, ref_sents)
    assert sari_score == pytest.approx(37.08210095744638)

    hyp_sents = read_lines("data/system_outputs/turk/lower/EncDecA.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, ref_sents)
    assert sari_score == pytest.approx(35.65754396121206)

    hyp_sents = read_lines("data/system_outputs/turk/lower/Hybrid.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, ref_sents)
    assert sari_score == pytest.approx(31.39665078989411)

    hyp_sents = read_lines("data/system_outputs/turk/lower/PBMT-R.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, ref_sents)
    assert sari_score == pytest.approx(38.558843050332037)

    hyp_sents = read_lines("data/system_outputs/turk/lower/SBMT-SARI.tok.low")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, ref_sents)
    assert sari_score == pytest.approx(39.964857928109127)

