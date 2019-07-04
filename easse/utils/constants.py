from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent.parent.parent
TOOLS_DIR = REPO_DIR / 'tools'
DATA_DIR = REPO_DIR / 'data'
CONFIG_PATH = REPO_DIR / 'easse/config.json'
STANFORD_CORENLP_DIR = TOOLS_DIR / 'stanford-corenlp-full-2018-10-05'
UCCA_DIR = TOOLS_DIR / 'ucca-bilstm-1.3.10'
UCCA_PARSER_PATH = UCCA_DIR / 'models/ucca-bilstm'
