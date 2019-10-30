# EASSE
[**EASSE**: **E**asier **A**utomatic **S**entence **S**implification **E**valuation](https://arxiv.org/abs/1908.04567)

## Installation
<!--
### Installing via pip

   ```bash
   pip install easse
   ```
-->
### Installing from source

Install EASSE by running:

```bash
git clone https://github.com/feralvam/easse.git
cd easse
pip install .
```

This will make `easse` available on your system but it will use the sources from the local clone
you made of the source repository.

## Running EASSE

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

### evaluate
```
$ easse evaluate -h
Usage: easse evaluate [OPTIONS]

  Evaluate a system output with automatic metrics.

Options:
  -t, --test_set [turk|turk_valid|pwkp|hsplit]
                                  test set to use.  [required]
  -tok, --tokenizer [13a|intl|moses|plain]
                                  Tokenization method to use.
  -m, --metrics TEXT              Comma-separated list of metrics to compute.
                                  Default: bleu,sari,samsa,fkgl
  -a, --analysis                  Perform word-level transformation analysis.
  -q, --quality_estimation        Perform quality estimation.
  -h, --help                      Show this message and exit.
```
Example with the [ACCESS](https://github.com/facebookresearch/access) system outputs:
```
easse evaluate -t turk -m 'bleu,sari' -q < easse/resources/data/system_outputs/turk/lower/ACCESS.tok.low
```

<img src="https://github.com/feralvam/easse/blob/master/demo/evaluate.gif">

### report
```
$ easse report -h
Usage: easse report [OPTIONS]

  Create a HTML report file with automatic metrics, plots and samples.

Options:
  -t, --test_set [turk|turk_valid|pwkp|hsplit]
                                  test set to use.  [required]
  -tok, --tokenizer [13a|intl|moses|plain]
                                  Tokenization method to use.
  -p, --report_path PATH          Path to the output HTML report.
  -h, --help                      Show this message and exit.
```
Example with the [ACCESS](https://github.com/facebookresearch/access) system outputs:
```
easse report -t turk < easse/resources/data/system_outputs/turk/lower/ACCESS.tok.low
```
<img src="https://github.com/feralvam/easse/blob/master/demo/report.gif">

## Citation
If you use EASSE in your research, please cite [EASSE: Easier Automatic Sentence Simplification Evaluation](https://arxiv.org/abs/1908.04567)

```
@inproceedings{alvamanchego-etal:2019:easse,
    title = "{EASSE}: {E}asier {A}utomatic {S}entence {S}implification {E}valuation",
    author = "Alva-Manchego, Fernando and Martin, Louis and Scarton, Carolina and Specia, Lucia",
    booktitle = "To Appear in EMNLP-ICJNLP 2019: System Demonstrations",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    url = "https://arxiv.org/abs/1908.04567",
}
```
