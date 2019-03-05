import os
from pathlib import Path
import tempfile
import unittest.mock

import nltk
from tupa.config import Config
from tupa.parse import read_passages, Parser

from samsa.utils import RESOURCES_DIR


def get_ucca_parser():
    ucca_dir = RESOURCES_DIR / 'ucca'
    os.chdir(str(ucca_dir))
    model_path = ucca_dir / 'models/ucca-bilstm'
    vocab_path = ucca_dir / 'vocab'
    argv = ['script_name', '-m', str(model_path), '--vocab', str(vocab_path)]
    with unittest.mock.patch('sys.argv', argv):
        Config.reload()
        args = Config().args
    model_files = [base + '' + ext for base, ext in map(os.path.splitext, args.models or (args.classifier,))]
    return Parser(model_files=model_files, config=Config(), beam=1)


PARSER = get_ucca_parser()


def get_ucca_passage(sentence):
    source_path = Path(tempfile.mkdtemp()) / 'source.txt'
    with source_path.open('w') as f:
        f.write(sentence + '\n')
    argv = ['script_name', str(source_path)]
    with unittest.mock.patch('sys.argv', argv):
        Config.reload()
        args = Config().args
    train_passages, dev_passages, test_passages = [read_passages(args, arg) for arg in
                                                   (args.train, args.dev, args.passages)]
    ucca_passages = [ucca_passage for (ucca_passage,) in PARSER.parse(test_passages,
                                                                      evaluate=[],
                                                                      display=False,
                                                                      write=False)]
    assert len(ucca_passages) == 1
    return ucca_passages[0]


def get_scenes(text):
    """Return all the ucca scenes in the given text"""
    ucca_passage = get_ucca_passage(text)
    ucca_scenes = [x for x in ucca_passage.layer('1').all if x.tag == "FN" and x.is_scene()]
    text_scenes = []
    for scene in ucca_scenes:
        words = []
        previous_word = ''
        for terminal in scene.get_terminals(False, True):
            word = terminal.text
            if word == previous_word:
                # TODO: Iterating this way on the scene sometimes yields duplicates.
                continue
            words.append(word)
            previous_word = word
        text_scenes.append(words)
    return text_scenes


def get_sentences(text):
    """Splits the input text into its sentences and each output sentence to words."""
    word_tokenizer = nltk.tokenize.WordPunctTokenizer()
    # TODO: Maybe we should remove punctuation as well to match tupa's word tokenization
    sentence_tokenizer = nltk.data.load(f'tokenizers/punkt/english.pickle')
    return [word_tokenizer.tokenize(sentence)
            for sentence in sentence_tokenizer.tokenize(text)]
