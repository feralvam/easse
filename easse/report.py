from collections import OrderedDict
from typing import List

from sacrebleu import corpus_bleu
from yattag import Doc, indent

from easse.fkgl import corpus_fkgl
from easse.quality_estimation import corpus_quality_estimation
from easse.samsa import corpus_samsa
from easse.sari import corpus_sari
from easse.utils.helpers import add_dicts


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


def add_html_head(doc):
    doc.asis('''
  <head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
    <!-- Solarized CSS -->
    <link href="https://codepen.io/louismartincs/pen/mZqjLG.css" rel="stylesheet"></link>

  </head>
''') # noqa
    return doc


def add_scores_table(doc, scores):
    def add_row(doc, values, tag_type='td'):
        with doc.tag('tr'):
            for value in values:
                with doc.tag(tag_type):
                    doc.text(value)

    with doc.tag('table', klass='table table-bordered table-responsive'):
        # Header
        with doc.tag('thead', klass='thead-light'):
            add_row(doc, scores.keys(), tag_type='th')
        # Values
        with doc.tag('tbody'):
            add_row(doc, [round(score, 2) for score in scores.values()], tag_type='td')
    return doc


def get_html_report(orig_sents: List[str], sys_sents: List[str], refs_sents: List[List[str]],
                    lowercase: bool = False, tokenizer: str = '13a'):
    doc = Doc()
    doc.asis('<!doctype html>')
    with doc.tag('html', lang='en'):
        doc = add_html_head(doc)
        with doc.tag('div', klass='container-fluid'):
            with doc.tag('h1'):
                doc.text('EASSE report')
            scores = get_all_scores(orig_sents, sys_sents, refs_sents,
                                    lowercase=False, tokenizer='13a')
            doc = add_scores_table(doc, scores)
    return indent(doc.getvalue())


def write_html_report(filepath, *args, **kwargs):
    with open(filepath, 'w') as f:
        f.write(get_html_report(*args, **kwargs) + '\n')
