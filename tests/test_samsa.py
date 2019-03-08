import pytest
import easse.samsa.samsa_score as samsa
from easse.samsa.ucca_utils import get_scenes


def test_get_scenes():
    text = 'You are waiting for a train , this train will take you far away .'
    expected_scenes = [
        ['You', 'are', 'waiting', 'for', 'a', 'train'],
        ['this', 'train', 'will', 'take', 'you', 'far', 'away']
    ]
    scenes = get_scenes(text)
    print(scenes)
    assert scenes == expected_scenes


# def test_samsa_score():
#     orig_sentence = "You are waiting for a train , this train will take you far away ."
#     sys_sentence = "You are waiting for a train . A train that will take you far away ."
#
#     samsa_score = samsa.samsa_sentence(orig_sentence, sys_sentence)
#
#     assert samsa_score == pytest.approx(1.0)
