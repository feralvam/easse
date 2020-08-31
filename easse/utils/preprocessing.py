from typing import List

import sacrebleu
import sacremoses


def normalize(sentence, lowercase: bool = True, tokenizer: str = '13a', return_str: bool = True):
    if lowercase:
        sentence = sentence.lower()

    if tokenizer in ['13a', 'intl']:
        normalized_sent = sacrebleu.TOKENIZERS[tokenizer]()(sentence)
    elif tokenizer == 'moses':
        normalized_sent = sacremoses.MosesTokenizer().tokenize(sentence, return_str=True, escape=False)
    elif tokenizer == 'penn':
        normalized_sent = sacremoses.MosesTokenizer().penn_tokenize(sentence, return_str=True)
    else:
        normalized_sent = sentence

    if not return_str:
        normalized_sent = normalized_sent.split()

    return normalized_sent
