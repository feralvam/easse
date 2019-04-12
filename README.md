# EASSE
Easier Automatic Sentence Simplification Evaluation

## Installation
<!--
### Installing via pip

   ```bash
   pip install easse
   ```
-->
### Installing from source

You can also install EASSE by cloning our git repository:

```bash
git clone https://github.com/feralvam/easse.git
```

Create a Python 3.6 virtual environment, and install EASSE in `editable` mode by running:

```bash
pip install --editable .
```

This will make `easse` available on your system but it will use the sources from the local clone
you made of the source repository.

## Running EASSE

Once EASSE has been installed, you can run the command-line interface with the `easse` command.

```bash
$ easse
Usage: easse [OPTIONS] COMMAND [ARGS]...

Options:
  --version   Show the version and exit.
  -h, --help  Show this message and exit.

Commands:
  evaluate  Evaluate a system output with automatic metrics.
  ranking   Rank all available system outputs in a standard test set.
  register  Preprocess and store a test set locally.
```