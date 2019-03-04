import pytest
import easse.sari as sari


def read_file(filename):
    with open(filename, encoding="utf8") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


# def test_sari_sentence():
#     orig_sent = "About 95 species are currently accepted ."
#     ref_sents = ["About 95 species are currently known .",
#                  "About 95 species are now accepted .",
#                  "95 species are now accepted ."]
#
#     sys_sent = "About 95 you now get in ."
#     sari_score = sari.sari_sentence(orig_sent, sys_sent, ref_sents)
#     assert sari_score == pytest.approx(0.2682782411698074)
#
#     sys_sent = "About 95 species are now agreed ."
#     sari_score = sari.sari_sentence(orig_sent, sys_sent, ref_sents)
#     assert sari_score == pytest.approx(0.5889995423074248)
#
#     sys_sent = "About 95 species are currently agreed ."
#     sari_score = sari.sari_sentence(orig_sent, sys_sent, ref_sents)
#     assert sari_score == pytest.approx(0.5071608864657479)


def test_sari_corpus_plain():
    orig_sents = read_file("data/turkcorpus/test.8turkers.tok.norm")
    for n in range(8):
        ref_lines = read_file(f"data/turkcorpus/test.8turkers.tok.turk.{n}")
        if n == 0:
            ref_sents = [[line] for line in ref_lines]
        else:
            ref_sents = [x + [y] for x, y in zip(ref_sents, ref_lines)]

    hyp_sents = read_file("data/system_outputs/lower/Dress-Ls.lower")
    sari_score = sari.sari_corpus(orig_sents, hyp_sents, ref_sents, tokenizer='plain')
    assert sari_score == pytest.approx(0.3673586275692667)

    hyp_sents = read_file("data/system_outputs/lower/Dress.lower")
    sari_score = sari.sari_corpus(orig_sents, hyp_sents, ref_sents, tokenizer='plain')
    assert sari_score == pytest.approx(0.365859900146575)

    hyp_sents = read_file("data/system_outputs/lower/EncDecA.lower")
    sari_score = sari.sari_corpus(orig_sents, hyp_sents, ref_sents, tokenizer='plain')
    assert sari_score == pytest.approx(0.3473946658449856)

    hyp_sents = read_file("data/system_outputs/lower/Hybrid.lower")
    sari_score = sari.sari_corpus(orig_sents, hyp_sents, ref_sents, tokenizer='plain')
    assert sari_score == pytest.approx(0.31008109926854227)

    hyp_sents = read_file("data/system_outputs/lower/PBMT-R.lower")
    sari_score = sari.sari_corpus(orig_sents, hyp_sents, ref_sents, tokenizer='plain')
    assert sari_score == pytest.approx(0.37817966679481013)

    hyp_sents = read_file("data/system_outputs/lower/SBMT-SARI.lower")
    sari_score = sari.sari_corpus(orig_sents, hyp_sents, ref_sents, tokenizer='plain')
    assert sari_score == pytest.approx(0.39360477024519125)


def test_sari_corpus_tokenize():
    orig_sents = read_file("data/turkcorpus/test.8turkers.tok.norm")
    for n in range(8):
        ref_lines = read_file(f"data/turkcorpus/test.8turkers.tok.turk.{n}")
        if n == 0:
            ref_sents = [[line] for line in ref_lines]
        else:
            ref_sents = [x + [y] for x, y in zip(ref_sents, ref_lines)]

    hyp_sents = read_file("data/system_outputs/lower/Dress-Ls.lower")
    sari_score = sari.sari_corpus(orig_sents, hyp_sents, ref_sents)
    assert sari_score == pytest.approx(0.37266058818588216)

    hyp_sents = read_file("data/system_outputs/lower/Dress.lower")
    sari_score = sari.sari_corpus(orig_sents, hyp_sents, ref_sents)
    assert sari_score == pytest.approx(0.3708210095744638)

    hyp_sents = read_file("data/system_outputs/lower/EncDecA.lower")
    sari_score = sari.sari_corpus(orig_sents, hyp_sents, ref_sents)
    assert sari_score == pytest.approx(0.3565754396121206)

    hyp_sents = read_file("data/system_outputs/lower/Hybrid.lower")
    sari_score = sari.sari_corpus(orig_sents, hyp_sents, ref_sents)
    assert sari_score == pytest.approx(0.3139665078989411)

    hyp_sents = read_file("data/system_outputs/lower/PBMT-R.lower")
    sari_score = sari.sari_corpus(orig_sents, hyp_sents, ref_sents)
    assert sari_score == pytest.approx(0.38558843050332037)

    hyp_sents = read_file("data/system_outputs/lower/SBMT-SARI.lower")
    sari_score = sari.sari_corpus(orig_sents, hyp_sents, ref_sents)
    assert sari_score == pytest.approx(0.39964857928109127)

