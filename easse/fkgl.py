from typing import List

from easse.utils.preprocessing import normalize
from easse.utils.text import (
    to_sentences,
    count_words,
    count_syllables_in_sentence,
)


class FKGLScorer:
    "https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests"

    def __init__(self):
        self.nb_words = 0
        self.nb_syllables = 0
        self.nb_sentences = 0

    def add(self, text):
        for sentence in to_sentences(text):
            self.nb_words += count_words(sentence)
            self.nb_syllables += count_syllables_in_sentence(sentence)
            self.nb_sentences += 1

    def score(self):
        # Flesch-Kincaid grade level
        if self.nb_sentences == 0 or self.nb_words == 0:
            return 0
        return max(
            0,
            0.39 * (self.nb_words / self.nb_sentences) + 11.8 * (self.nb_syllables / self.nb_words) - 15.59,
        )


def corpus_fkgl(sentences: List[str], tokenizer: str = "13a"):
    scorer = FKGLScorer()
    for sentence in sentences:
        scorer.add(normalize(sentence, tokenizer=tokenizer))
    return scorer.score()
