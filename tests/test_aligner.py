import pytest

import easse.aligner.aligner as aligner


@pytest.mark.skip(reason="TODO: Aligner is currently broken")
def test_align_str():
    # aligning strings (output indexes start at 1)
    sentence1 = "Four men died in an accident."
    sentence2 = "4 people are dead from a collision."

    alignments = aligner.align(sentence1, sentence2)

    assert alignments[0] == [[7, 8], [2, 2], [3, 4], [1, 1], [6, 7], [5, 6]]
    assert alignments[1] == [
        ['.', '.'],
        ['men', 'people'],
        ['died', 'dead'],
        ['Four', '4'],
        ['accident', 'collision'],
        ['an', 'a'],
    ]


@pytest.mark.skip(reason="TODO: Aligner is currently broken")
def test_align_tokens():
    # aligning sets of tokens (output indexes start at 1)
    sentence1 = ['Four', 'men', 'died', 'in', 'an', 'accident', '.']
    sentence2 = ['4', 'people', 'are', 'dead', 'from', 'a', 'collision', '.']

    alignments = aligner.align(sentence1, sentence2)

    assert alignments[0] == [[7, 8], [2, 2], [3, 4], [1, 1], [6, 7], [5, 6]]
    assert alignments[1] == [
        ['.', '.'],
        ['men', 'people'],
        ['died', 'dead'],
        ['Four', '4'],
        ['accident', 'collision'],
        ['an', 'a'],
    ]
