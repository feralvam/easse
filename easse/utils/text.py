import re
from functools import lru_cache

import nltk


def to_words(text):
    return text.split()


def count_words(text):
    return len(to_words(text))


def to_sentences(text, language='english'):
    try:
        tokenizer = nltk.data.load(f'tokenizers/punkt/{language}.pickle')
    except LookupError:
        nltk.download('punkt')
        tokenizer = nltk.data.load(f'tokenizers/punkt/{language}.pickle')
    return tokenizer.tokenize(text)


def count_sentences(text, language='english'):
    return len(to_sentences(text, language))


@lru_cache(maxsize=100000)
def count_syllables_in_word(word):
    # The syllables counting logic is adapted from the following scripts:
    # https://github.com/XingxingZhang/dress/blob/master/dress/scripts/readability/syllables_en.py
    # https://github.com/nltk/nltk_contrib/blob/master/nltk_contrib/readability/syllables_en.py
    special_words = {
        'the': 1,
        'tottered': 2,
        'chummed': 1,
        'peeped': 1,
        'moustaches': 2,
        'shamefully': 3,
        'messieurs': 2,
        'satiated': 4,
        'sailmaker': 4,
        'sheered': 1,
        'disinterred': 3,
        'propitiatory': 6,
        'bepatched': 2,
        'particularized': 5,
        'caressed': 2,
        'trespassed': 2,
        'sepulchre': 3,
        'flapped': 1,
        'hemispheres': 3,
        'pencilled': 2,
        'motioned': 2,
        'poleman': 2,
        'slandered': 2,
        'sombre': 2,
        'etc': 4,
        'sidespring': 2,
        'mimes': 1,
        'effaces': 2,
        'mr': 2,
        'mrs': 2,
        'ms': 1,
        'dr': 2,
        'st': 1,
        'sr': 2,
        'jr': 2,
        'truckle': 2,
        'foamed': 1,
        'fringed': 2,
        'clattered': 2,
        'capered': 2,
        'mangroves': 2,
        'suavely': 2,
        'reclined': 2,
        'brutes': 1,
        'effaced': 2,
        'quivered': 2,
        "h'm": 1,
        'veriest': 3,
        'sententiously': 4,
        'deafened': 2,
        'manoeuvred': 3,
        'unstained': 2,
        'gaped': 1,
        'stammered': 2,
        'shivered': 2,
        'discoloured': 3,
        'gravesend': 2,
        '60': 2,
        'lb': 1,
        'unexpressed': 3,
        'greyish': 2,
        'unostentatious': 5,
    }
    special_syllables_substract = ['cial', 'tia', 'cius', 'cious', 'gui', 'ion', 'iou', 'sia$', '.ely$']
    special_syllables_add = [
        'ia',
        'riet',
        'dien',
        'iu',
        'io',
        'ii',
        '[aeiouy]bl$',
        'mbl$',
        '[aeiou]{3}',
        '^mc',
        'ism$',
        '(.)(?!\\1)([aeiouy])\\2l$',
        '[^l]llien',
        '^coad.',
        '^coag.',
        '^coal.',
        '^coax.',
        '(.)(?!\\1)[gq]ua(.)(?!\\2)[aeiou]',
        'dnt$',
    ]
    word = word.lower().strip()
    if word in special_words:
        return special_words[word]
    # Remove final silent 'e'
    word = word.rstrip('e')
    # Count vowel groups
    count = 0
    prev_was_vowel = 0
    for c in word:
        is_vowel = c in ('a', 'e', 'i', 'o', 'u', 'y')
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel

    # Add & subtract syllables
    for r in special_syllables_add:
        if re.search(r, word):
            count += 1
    for r in special_syllables_substract:
        if re.search(r, word):
            count -= 1
    return count


def count_syllables_in_sentence(sentence):
    return sum([count_syllables_in_word(word) for word in to_words(sentence)])
