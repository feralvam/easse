from easse.samsa.scene_sentence_extraction import get_sentences, get_scenes


def test_get_sentences():
    text = 'You are waiting for a train . A train that will take you far away .'
    expected_sentences = [
        ['You', 'are', 'waiting', 'for', 'a', 'train', '.'],
        ['A', 'train', 'that', 'will', 'take', 'you', 'far', 'away', '.'],
    ]
    sentences = get_sentences(text)
    assert sentences == expected_sentences


def test_get_scenes():
    text = 'You are waiting for a train , this train will take you far away .'
    expected_scenes = [
        ['You', 'are', 'waiting', 'for', 'a', 'train'],
        ['this', 'train', 'that', 'will', 'take', 'you', 'far', 'away']
    ]
    scenes = get_scenes(text)
    print(scenes)
    assert scenes == expected_scenes
