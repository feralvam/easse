from easse.utils.preprocessing import split_into_sentences


def test_split_into_sentences():
    text = 'You are waiting for a train . A train that will take you far away .'
    expected_sentences = [
        ['you', 'are', 'waiting', 'for', 'a', 'train', '.'],
        ['a', 'train', 'that', 'will', 'take', 'you', 'far', 'away', '.'],
    ]
    sentences = split_into_sentences(text, normalized=False, return_str=False)
    print(sentences)
    assert sentences == expected_sentences
