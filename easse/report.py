from collections import OrderedDict
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sacrebleu import corpus_bleu
import seaborn as sns
from tseval.feature_extraction import get_levenshtein_similarity, get_compression_ratio, count_sentence_splits
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
        formatted_string = ''
        for mutual_word in mutual_words:
            word = next(words_generator)
            bold_text = ''
            while word != mutual_word:
                bold_text += ' ' + word
                word = next(words_generator)
            if bold_text != '':
                formatted_string += ' ' + make_bold(bold_text)
            formatted_string += ' ' + word
        # Add remaining words
        formatted_string += make_bold(' '.join(words_generator))
        return formatted_string.strip()

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
        ('Wikilarge predictions with the lowest Levenshtein similarity',
         lambda c, s: get_levenshtein_similarity(c, s)),
    ]
    doc = Doc()
    for title, sort_key in title_key:
        with doc.tag('div', klass='container-fluid mt-5'):
            doc.line('h3', title)
            doc.stag('hr')
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


def save_histogram(sys_values, ref_values, filepath, title, xmax=1.5):
    step = 0.02
    linewidth = 0.9
    plt.hist(sys_values, bins=np.arange(0, 1.5, step), density=True, label='System output', color='firebrick',
             linewidth=linewidth, alpha=0.8)
    plt.hist(ref_values, bins=np.arange(0, 1.5, step), density=True, label='Reference', color='forestgreen',
             linewidth=linewidth, alpha=0.4)
    plt.legend()
    plt.title(title)
    plt.xlim(0, xmax)
    plt.ylim(0, 10)
    plt.xticks(np.arange(0, xmax + 0.1, 0.25), [f'{int(ratio*100)}%' for ratio in np.arange(0, 1.51, 0.25)])
    plt.savefig(filepath)
    plt.clf()


def get_compression_ratios_plot_html(orig_sents, sys_sents, ref_sents, plots_dirpath):
    doc = Doc()
    pred_values = []
    ref_values = []
    for orig_sent, sys_sent, ref_sent in zip(orig_sents, sys_sents, ref_sents):
        pred_values.append(get_compression_ratio(orig_sent, sys_sent))
        ref_values.append(get_compression_ratio(orig_sent, ref_sent))
    filepath = plots_dirpath / 'compression_ratios.png'
    title = 'Compression ratios'
    save_histogram(pred_values, ref_values, filepath=filepath, title=title, xmax=1.5)
    doc.stag('img', src=str(filepath))
    return doc.getvalue()


def get_levenshtein_similarity_plot_html(orig_sents, sys_sents, ref_sents, plots_dirpath):
    doc = Doc()
    pred_values = []
    ref_values = []
    for orig_sent, sys_sent, ref_sent in zip(orig_sents, sys_sents, ref_sents):
        pred_values.append(get_levenshtein_similarity(orig_sent, sys_sent))
        ref_values.append(get_levenshtein_similarity(orig_sent, ref_sent))
    filepath = plots_dirpath / 'levenshtein_similarity.png'
    title = 'Levenshtein similarity'
    save_histogram(pred_values, ref_values, filepath=filepath, title=title, xmax=1)
    doc.stag('img', src=str(filepath))
    return doc.getvalue()


def get_plots_html(orig_sents, sys_sents, ref_sents, plots_dirpath):
    doc = Doc()
    plots_dirpath = Path(plots_dirpath)
    plots_dirpath.mkdir(exist_ok=True)
    doc.asis(get_compression_ratios_plot_html(orig_sents, sys_sents, ref_sents, plots_dirpath))
    doc.asis(get_levenshtein_similarity_plot_html(orig_sents, sys_sents, ref_sents, plots_dirpath))
    return doc.getvalue()


def get_scores_by_length_html(orig_sents, sys_sents, refs_sents, n_bins=5):
    def get_intervals_from_limits(limits):
        return list(zip(limits[:-1], limits[1:]))

    def get_equally_populated_intervals(sents, n_bins):
        sent_lengths = sorted([len(sent) for sent in sents])
        n_samples_per_bin = int(len(sent_lengths) / n_bins)
        limits = [sent_lengths[i * n_samples_per_bin] for i in range(n_bins)] + [sent_lengths[-1] + 1]
        return get_intervals_from_limits(limits)

    def split_sents_by_lengths(sents, intervals):
        bins = [[] for _ in range(len(intervals))]
        for sent_idx, sent in enumerate(sents):
            sent_length = len(sent)
            for interval_idx, (interval_start, interval_end) in enumerate(intervals):
                if interval_start <= sent_length and sent_length < interval_end:
                    bins[interval_idx].append(sent_idx)
                    break
        assert sum([len(b) for b in bins]) == len(sents)
        return bins

    def split_sents_by_bins(sents, bins):
        sents = np.array(sents)
        sents_by_bin = []
        for sent_indexes in bins:
            sents_by_bin.append(np.array(sents)[sent_indexes])
        return sents_by_bin

    def df_append_row(df, row, row_name=None):
        if row_name is None:
            return df.append(pd.Series(row), ignore_index=True)
        else:
            return df.append(pd.Series(row, name=row_name))

    intervals = get_equally_populated_intervals(orig_sents, n_bins)
    bins = split_sents_by_lengths(orig_sents, intervals)
    # Split files by bins
    orig_sents_by_bins = split_sents_by_bins(orig_sents, bins)
    sys_sents_by_bins = split_sents_by_bins(sys_sents, bins)
    refs_sents_by_bins = [split_sents_by_bins(ref_sents, bins) for ref_sents in refs_sents]
    df_bins = pd.DataFrame()
    # Get scores for each bin
    for i in range(len(intervals)):
        interval = intervals[i]
        splitted_orig_sents = orig_sents_by_bins[i]
        splitted_sys_sents = sys_sents_by_bins[i]
        splitted_refs_sents = [ref_sents_by_bins[i] for ref_sents_by_bins in refs_sents_by_bins]
        scores = get_all_scores(splitted_orig_sents, splitted_sys_sents, splitted_refs_sents)
        row_name = f'length=[{interval[0]};{interval[1]}]'
        df_bins = df_append_row(df_bins, scores, row_name)
    html = df_bins.round(2).to_html(classes='table table-bordered table-responsive table-striped')
    return html.replace('<thead>', '<thead class="thead-light">')


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
    with doc.tag('table', klass='table table-bordered table-responsive table-striped'):
        with doc.tag('thead', klass='thead-light'):
            add_header(doc, header)
        with doc.tag('tbody'):
            for row, row_name in zip(rows, row_names):
                add_row(doc, [round(val, 2) for val in row], row_name)
    return doc.getvalue()


def get_html_report(orig_sents: List[str], sys_sents: List[str], refs_sents: List[List[str]], plots_dirpath=None,
                    lowercase: bool = False, tokenizer: str = '13a'):
    sns.set_style('darkgrid')
    doc = Doc()
    doc.asis('<!doctype html>')
    with doc.tag('html', lang='en'):
        doc.asis(get_head_html())
        with doc.tag('div', klass='container-fluid'):
            doc.line('h1', 'EASSE report')
            doc.stag('hr')

            doc.line('h2', 'Scores')
            doc.stag('hr')
            with doc.tag('div', klass='container-fluid'):
                doc.line('h3', 'System vs. Reference')
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
                doc.line('h3', 'By sentence length (characters)')
                doc.stag('hr')
                doc.asis(get_scores_by_length_html(orig_sents, sys_sents, refs_sents))
            doc.line('h2', 'Plots')
            doc.stag('hr')
            with doc.tag('div', klass='container-fluid'):
                if plots_dirpath is None:
                    plots_dirpath = Path.cwd()
                doc.asis(get_plots_html(orig_sents, sys_sents, refs_sents[0], plots_dirpath))
            doc.line('h2', 'Qualitative evaluation')
            doc.stag('hr')
            with doc.tag('div', klass='container-fluid'):
                doc.asis(get_qualitative_html_examples(orig_sents, sys_sents))
    return indent(doc.getvalue())


def write_html_report(filepath, *args, **kwargs):
    with open(filepath, 'w') as f:
        plots_dirpath = Path(filepath).parent / 'html_plots'
        f.write(get_html_report(*args, plots_dirpath=plots_dirpath, **kwargs) + '\n')
