# EASSE
[**EASSE**](https://www.aclweb.org/anthology/D19-3009/) (**E**asier **A**utomatic **S**entence **S**implification **E**valuation) is a Python 3 package aiming to facilitate and standardise automatic evaluation and comparison of Sentence Simplification systems. ([*What is Sentence Simplification?*](https://nlpprogress.com/english/simplification.html))

### Features

- Automatic evaluation metrics (e.g. SARI, BLEU, SAMSA, etc.)
- Word-level transformation analysis 
- Referenceless Quality Estimation features
- Straightforward access to commonly used evaluation datasets
- Comprehensive HTML report for quantitative and qualitative evaluation of a simplification output

## Installation
### Requirements

Python 3.6 or 3.7 is required.

### Installing from Source

Install EASSE by running:

```
git clone https://github.com/feralvam/easse.git
cd easse
pip install .
```

This will make `easse` available on your system but it will use the sources from the local clone
you made of the source repository.

## Running EASSE

### CLI
Once EASSE has been installed, you can run the command-line interface with the `easse` command.

```
$ easse
Usage: easse [OPTIONS] COMMAND [ARGS]...

Options:
  --version   Show the version and exit.
  -h, --help  Show this message and exit.

Commands:
  evaluate  Evaluate a system output with automatic metrics.
  report    Create a HTML report file with automatic metrics, plots and samples.
```

#### easse evaluate
```
$ easse evaluate -h
Usage: easse evaluate [OPTIONS]

Options:
  -m, --metrics TEXT              Comma-separated list of metrics to compute.
                                  Valid: bleu,sari,samsa,fkgl (SAMSA is
                                  disabled by default for the sake of speed)
  -tok, --tokenizer [13a|intl|moses|plain]
                                  Tokenization method to use.
  --refs_sents_paths TEXT         Comma-separated list of path(s) to the
                                  references(s). Only used when test_set ==
                                  "custom"
  --orig_sents_path PATH          Path to the source sentences. Only used when
                                  test_set == "custom"
  --sys_sents_path PATH           Path to the system predictions input file
                                  that is to be evaluated.
  -t, --test_set [turk|turk_valid|pwkp|pwkp_valid|hsplit|custom]
                                  test set to use.  [required]
  -a, --analysis                  Perform word-level transformation analysis.
  -q, --quality_estimation        Perform quality estimation.
  -h, --help                      Show this message and exit.
```
Example with the [ACCESS](https://github.com/facebookresearch/access) system outputs:
```
easse evaluate -t turk -m 'bleu,sari' -q < easse/resources/data/system_outputs/turk/lower/ACCESS.tok.low
```

<img src="https://github.com/feralvam/easse/blob/master/demo/evaluate.gif">

#### easse report
```
$ easse report -h
Usage: easse report [OPTIONS]

Options:
  -m, --metrics TEXT              Comma-separated list of metrics to compute.
                                  Valid: bleu,sari,samsa,fkgl (SAMSA is
                                  disabled by default for the sake of speed
  -tok, --tokenizer [13a|intl|moses|plain]
                                  Tokenization method to use.
  --refs_sents_paths TEXT         Comma-separated list of path(s) to the
                                  references(s). Only used when test_set ==
                                  "custom"
  --orig_sents_path PATH          Path to the source sentences. Only used when
                                  test_set == "custom"
  --sys_sents_path PATH           Path to the system predictions input file
                                  that is to be evaluated.
  -t, --test_set [turk|turk_valid|pwkp|pwkp_valid|hsplit|custom]
                                  test set to use.  [required]
  -p, --report_path PATH          Path to the output HTML report.
  -h, --help                      Show this message and exit.
```
Example:
```
easse report -t turk < easse/resources/data/system_outputs/turk/lower/ACCESS.tok.low
```
<img src="https://github.com/feralvam/easse/blob/master/demo/report.gif">

### Python

You can also use the different functions available in EASSE from your Python code.

```python
>>> from easse.sari import corpus_sari
>>> orig_sents = ["About 95 species are currently accepted ."]
>>> sys_sents = ["About 95 you now get in ."]
>>> refs_sents = [["About 95 species are currently known .", 
...                "About 95 species are now accepted .", 
...                "95 species are now accepted ."]]
>>> sari_score = corpus_sari(orig_sents, sys_sents, refs_sents)
>>> print(sari_score)
27.301587301587304
```

## Licence
EASSE is licenced under the GNU General Public License v3.0.

## Citation

If you use EASSE in your research, please cite [EASSE: Easier Automatic Sentence Simplification Evaluation](https://www.aclweb.org/anthology/D19-3009/)

```
@inproceedings{alva-manchego-etal-2019-easse,
    title = "{EASSE}: Easier Automatic Sentence Simplification Evaluation",
    author = "Alva-Manchego, Fernando  and
      Martin, Louis  and
      Scarton, Carolina  and
      Specia, Lucia",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP): System Demonstrations",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-3009",
    pages = "49--54"
}
```
