# EASSE
[**EASSE**](https://www.aclweb.org/anthology/D19-3009/) (**E**asier **A**utomatic **S**entence **S**implification **E**valuation) is a Python 3 package aiming to facilitate and standardise automatic evaluation and comparison of Sentence Simplification systems. ([*What is Sentence Simplification?*](https://www.mitpressjournals.org/doi/full/10.1162/coli_a_00370))

### Features

- Automatic evaluation metrics (e.g. SARI<sup>1</sup>, BLEU, SAMSA, etc.).
- Commonly used [**evaluation sets**](https://github.com/feralvam/easse/tree/master/easse/resources/data/test_sets).
- Literature [**system outputs**](https://github.com/feralvam/easse/tree/master/easse/resources/data/system_outputs) to compare to.
- Word-level transformation analysis.
- Referenceless Quality Estimation features.
- Straightforward access to commonly used evaluation datasets.
- Comprehensive HTML report for quantitative and qualitative evaluation of a simplification output.

[1]: The SARI version in EASSE fixes inconsistencies and bugs in the original version. See the dedicated section for more details.

## Installation
### Requirements

Python 3.6 or 3.7 is required.

### Installing from Source

Install EASSE by running:

```
git clone https://github.com/feralvam/easse.git
cd easse
pip install -e .
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
  -t, --test_set [turkcorpus_test|turkcorpus_valid|pwkp_test|pwkp_valid|hsplit_test|custom]
                                  test set to use.  [required]
  -a, --analysis                  Perform word-level transformation analysis.
  -q, --quality_estimation        Perform quality estimation.
  -h, --help                      Show this message and exit.
```
Example with the [ACCESS](https://github.com/facebookresearch/access) system outputs:
```
easse evaluate -t turkcorpus_test -m 'bleu,sari,fkgl' -q < easse/resources/data/system_outputs/turkcorpus/test/ACCESS
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
  -t, --test_set [turkcorpus_test|turkcorpus_valid|pwkp_test|pwkp_valid|hsplit_test|custom]
                                  test set to use.  [required]
  -p, --report_path PATH          Path to the output HTML report.
  -h, --help                      Show this message and exit.
```
Example:
```
easse report -t turkcorpus_test < easse/resources/data/system_outputs/turkcorpus/test/ACCESS
```
<img src="https://github.com/feralvam/easse/blob/master/demo/report.gif">

### Python

You can also use the different functions available in EASSE from your Python code.

```python
from easse.sari import corpus_sari

corpus_sari(orig_sents=["About 95 species are currently accepted.", "The cat perched on the mat."],  
            sys_sents=["About 95 you now get in.", "Cat on mat."], 
            refs_sents=[["About 95 species are currently known.", "The cat sat on the mat."],
                        ["About 95 species are now accepted.", "The cat is on the mat."],  
                        ["95 species are now accepted.", "The cat sat."]])
Out[1]: 33.17472563619544
```
## Differences with original SARI implementation

The version of SARI fixes inconsistencies and bugs that were present in the original implementation. The main differences are:
1) The original SARI implementation applies normalisation (NIST style tokenization and rejoin ‘s, ‘re ...) only on the prediction and references but not on the source sentence (see STAR.java file). This results in incorrect ngram additions or deletions. EASSE applies the same normalization to source, prediction and references.
2) The original SARI implementation takes tokenized text as input that are then tokenized a second time. This also causes discrepancies between the tokenization of the training set and the evaluation set. EASSE uses untokenized text that is then tokenized uniformly at runtime, during evaluation. This allows for training models on raw text without worrying about matching the evaluation tokenizer.
3) The original JAVA implementation had a silent overflow bug where ngram statistics would go beyond the maximum limit for integers and silently start over from the minimum value. This caused incorrect SARIs when rating too many sentences but did not raise an error.

## Licence
EASSE is licenced under the GNU General Public License v3.0.

## Citation

If you use EASSE in your research, please cite [EASSE: Easier Automatic Sentence Simplification Evaluation](https://aclanthology.org/D19-3009/)

```
@inproceedings{alva-manchego-etal-2019-easse,
    title = "{EASSE}: {E}asier Automatic Sentence Simplification Evaluation",
    author = "Alva-Manchego, Fernando  and
      Martin, Louis  and
      Scarton, Carolina  and
      Specia, Lucia",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP): System Demonstrations",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-3009",
    doi = "10.18653/v1/D19-3009",
    pages = "49--54",
}
```
