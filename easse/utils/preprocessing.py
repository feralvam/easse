import sacrebleu
import sacremoses
import os
from stanfordnlp.server import CoreNLPClient


os.environ['CORENLP_HOME'] = "/tools/stanford-corenlp-full-2018-10-05"


def normalize(sentence, lowercase: bool = True, tokenizer: str = '13a', return_str: bool = True):
    if lowercase:
        sentence = sentence.lower()

    if tokenizer == "13a":
        normalized_sent = sacrebleu.tokenize_13a(sentence)
    elif tokenizer == "intl":
        normalized_sent = sacrebleu.tokenize_v14_international(sentence)
    elif tokenizer == "moses":
        normalized_sent = sacremoses.MosesTokenizer().tokenize(sentence, return_str=True)
    else:
        normalized_sent = sentence

    if not return_str:
        normalized_sent = normalized_sent.split()

    return normalized_sent


def split_into_sentences(text, normalized=True, return_str: bool = False):
    if not normalized:
        text = normalize(text)

    with CoreNLPClient(annotators=['tokenize','ssplit'], properties={'tokenize.whitespace': True}) as client:
        ann = client.annotate(text)
        if return_str:
            sentences = [' '.join([t.word for t in s.token]) for s in ann.sentence]
        else:
            sentences = [[t.word for t in s.token] for s in ann.sentence]

    return sentences
