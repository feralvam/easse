from collections import OrderedDict
from typing import List

from sacrebleu import corpus_bleu
from tseval.feature_extraction import get_levenshtein_distance, get_compression_ratio, count_sentence_splits
from yattag import Doc, indent

from easse.fkgl import corpus_fkgl
from easse.quality_estimation import corpus_quality_estimation
from easse.samsa import corpus_samsa
from easse.sari import corpus_sari
from easse.utils.helpers import add_dicts
from easse.utils.text import to_words
from easse.annotation.lcs import get_lcs


def get_all_scores(orig_sents: List[str], sys_sents: List[str], refs_sents: List[List[str]],
                   lowercase: bool = False, tokenizer: str = '13a'):
    scores = OrderedDict()
    scores['BLEU'] = corpus_bleu(sys_sents, refs_sents, force=True, tokenize=tokenizer, lowercase=lowercase).score
    scores['SARI'] = corpus_sari(orig_sents, sys_sents, refs_sents, tokenizer=tokenizer, lowercase=lowercase)
    scores['SAMSA'] = corpus_samsa(orig_sents, sys_sents, tokenizer=tokenizer, verbose=True, lowercase=lowercase)
    scores['FKGL'] = corpus_fkgl(sys_sents, tokenizer=tokenizer)
    quality_estimation_scores = corpus_quality_estimation(
            orig_sents,
            sys_sents,
            tokenizer=tokenizer,
            lowercase=lowercase
            )
    return add_dicts(
            scores,
            quality_estimation_scores,
            )


def make_differing_words_bold(orig_sent, sys_sent, make_bold):
    '''Returns the two sentences with differing words in bold'''

    def format_words(words, mutual_words):
        '''Makes all words bold except the mutual ones'''
        words_generator = iter(words)
        formatted_words = []
        for mutual_word in mutual_words:
            word = next(words_generator)
            while word != mutual_word:
                formatted_words.append(make_bold(word))
                word = next(words_generator)
            formatted_words.append(word)
        # Add remaining words
        formatted_words.extend([make_bold(word) for word in words_generator])
        return ' '.join(formatted_words)

    orig_words = to_words(orig_sent)
    sys_words = to_words(sys_sent)
    mutual_words = get_lcs(orig_words, sys_words)
    return format_words(orig_words, mutual_words), format_words(sys_words, mutual_words)


def make_text_bold_html(text):
    doc = Doc()
    doc.line('strong', text)
    return doc.getvalue()


def get_qualitative_html_examples(orig_sents, sys_sents):
    title_key = [
        ('Random Wikilarge predictions',
         lambda c, s: 0),
        ('Wikilarge predictions with the most sentence splits',
         lambda c, s: -count_sentence_splits(c, s)),
        ('Wikilarge predictions with the lowest compression ratio',
         lambda c, s: get_compression_ratio(c, s)),
        ('Wikilarge predictions with the highest Levenshtein distances',
         lambda c, s: -get_levenshtein_distance(c, s)),
    ]
    doc = Doc()
    doc.line('h2', 'Qualitative evaluation')
    for title, sort_key in title_key:
        doc.stag('hr')
        doc.line('h3', title)
        n_samples = 10
        pair_generator = sorted(zip(orig_sents, sys_sents), key=lambda args: sort_key(*args))
        for i, (orig_sent, sys_sent) in enumerate(pair_generator):
            if i >= n_samples:
                break
            orig_sent_bold, sys_sent_bold = make_differing_words_bold(orig_sent, sys_sent, make_text_bold_html)
            with doc.tag('p'):
                doc.asis(orig_sent_bold)
                doc.stag('br')
                doc.asis(sys_sent_bold)
    return doc.getvalue()


def get_head_html():
    return '''
  <head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
    <!-- Solarized CSS -->
    <link href="https://codepen.io/louismartincs/pen/mZqjLG.css" rel="stylesheet"></link>

  </head>
''' # noqa


def get_table_html(header, rows, row_names=None):
    def add_header(doc, header):
        with doc.tag('tr'):
            for value in header:
                doc.line('th', value)

    def add_row(doc, values, row_name=None):
        with doc.tag('tr'):
            if row_name is not None:
                doc.line('th', row_name)
            for value in values:
                doc.line('td', value)

    doc = Doc()
    if row_names is not None:
        header.insert(0, '')
    else:
        row_names = [None] * len(rows)
    with doc.tag('table', klass='table table-bordered table-responsive'):
        with doc.tag('thead', klass='thead-light'):
            add_header(doc, header)
        with doc.tag('tbody'):
            for row, row_name in zip(rows, row_names):
                add_row(doc, [round(val, 2) for val in row], row_name)
    return doc.getvalue()


def get_html_report(orig_sents: List[str], sys_sents: List[str], refs_sents: List[List[str]],
                    lowercase: bool = False, tokenizer: str = '13a'):
    doc = Doc()
    doc.asis('<!doctype html>')
    with doc.tag('html', lang='en'):
        doc.asis(get_head_html())
        with doc.tag('div', klass='container-fluid'):
            doc.line('h1', 'EASSE report')
            doc.stag('hr')
            sys_scores = get_all_scores(orig_sents, sys_sents, refs_sents,
                                        lowercase=False, tokenizer='13a')
            ref_scores = get_all_scores(orig_sents, refs_sents[0], refs_sents[1:],
                                        lowercase=False, tokenizer='13a')
            assert sys_scores.keys() == ref_scores.keys()
            doc.asis(get_table_html(
                    header=list(sys_scores.keys()),
                    rows=[sys_scores.values(), ref_scores.values()],
                    row_names=['System output', 'Reference'],
                    ))
            doc.stag('hr')
            doc.asis(get_qualitative_html_examples(orig_sents, sys_sents))
    return indent(doc.getvalue())


def write_html_report(filepath, *args, **kwargs):
    with open(filepath, 'w') as f:
        f.write(get_html_report(*args, **kwargs) + '\n')
