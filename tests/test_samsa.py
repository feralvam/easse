import pytest
from easse.utils.ucca_utils import get_scenes_text, ucca_parse_texts
from easse.samsa import sentence_samsa, corpus_samsa
from easse.utils.resources import get_orig_sents, get_refs_sents


def test_get_scenes():
    text = 'You are waiting for a train , this train will take you far away .'
    expected_scenes = [
        ['You', 'are', 'waiting', 'for', 'a', 'train'],
        ['this', 'train', 'will', 'take', 'you', 'far', 'away'],
    ]
    scenes = get_scenes_text(ucca_parse_texts([text])[0])
    assert scenes == expected_scenes


def test_sentence_samsa():
    orig_sent = "You are waiting for a train, this train will take you far away."
    sys_sent = "You are waiting for a train. A train that will take you far away."
    samsa_score = sentence_samsa(orig_sent, sys_sent, lowercase=False, tokenizer='moses')
    assert samsa_score == pytest.approx(100.0)

    orig_sent = (
        "The river is indeed an ever-present part of the city's decor, and the official entrance "
        "to Lisbon is a broad marble stair mounting from the water to the vast, "
        "arcaded Commerce Square (Praca do Comercio)."
    )
    sys_sent = (
        "The river is indeed an ever-present part of the city's decor, and the entrance to Lisbon "
        "is a broad marble stair mounting from the water to the covering, arcaded Commerce "
        "Square (Praca do Comercio)."
    )
    samsa_score = sentence_samsa(orig_sent, sys_sent, lowercase=False, tokenizer='moses')
    assert samsa_score == pytest.approx(19.0)

    orig_sent = (
        "The second largest city of Russia and one of the world's major cities, "
        "St. Petersburg has played a vital role in Russian history."
    )
    sys_sent = (
        "The second largest city of Russia and one of the world's major cities, "
        "St Petersburg, and has played a vital role in Russian history."
    )
    samsa_score = sentence_samsa(orig_sent, sys_sent, lowercase=False, tokenizer='moses')
    assert samsa_score == pytest.approx(100.0)

    orig_sent = (
        "The incident followed the killing in August of five Egyptian security guards by "
        "Israeli soldiers pursuing militants who had ambushed and killed eight Israelis "
        "along the Israeli-Egyptian border."
    )
    sys_sent = (
        "The incident followed the killing in August by Israeli soldiers. "
        "Israeli soldiers pursued militants. "
        "Militants had ambushed and killed eight Israelis along the Israeli-Egyptian border."
    )
    samsa_score = sentence_samsa(orig_sent, sys_sent, lowercase=False, tokenizer='moses')
    assert samsa_score == pytest.approx(62.5)

    orig_sent = (
        "The injured man was able to drive his car to Cloverhill Prison where he got help. "
        "He is being treated at Tallaght Hospital but his injuries are not thought to be life-threatening."
    )
    sys_sent = "The injured man drive his car to Cloverhill Prison he got help."
    samsa_score = sentence_samsa(orig_sent, sys_sent, lowercase=False, tokenizer='moses')
    assert samsa_score == pytest.approx(14.0625)

    orig_sent = (
        "for example, king bhumibol was born on monday, so on his birthday throughout thailand will be "
        "decorated with yellow color."
    )
    sys_sent = (
        "for example, king bhumibol was born on monday, so on his birthday throughout thailand will be "
        "decorated with yellow color."
    )
    samsa_score = sentence_samsa(orig_sent, sys_sent, lowercase=False, tokenizer='moses')
    assert samsa_score == pytest.approx(50.0)


def test_corpus_samsa():
    orig_sents = get_orig_sents('qats_test')
    refs_sents = get_refs_sents('qats_test')
    samsa_score = corpus_samsa(orig_sents, refs_sents[0], lowercase=False, tokenizer='moses')
    assert samsa_score == pytest.approx(36.94996509406232)
