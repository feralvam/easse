import json
import os
import sys
import tarfile
import time
from urllib.request import urlretrieve
import warnings
import zipfile

from easse.utils.constants import (
    STANFORD_CORENLP_DIR,
    UCCA_DIR,
    UCCA_PARSER_PATH,
    TEST_SETS_PATHS,
    SYSTEM_OUTPUTS_DIRS_MAP,
)
from easse.utils.helpers import get_temp_filepath, read_lines, safe_divide

def reporthook(count, block_size, total_size):
    # Download progress bar
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size_mb = count * block_size / (1024 * 1024)
    speed = safe_divide(progress_size_mb, duration)
    percent = int(count * block_size * 100 / total_size)
    msg = f'\r... {percent}% - {int(progress_size_mb)} MB - {speed:.2f} MB/s - {int(duration)}s'
    sys.stdout.write(msg)


def download(url, destination_path):
    print(f'Downloading {url}...')
    try:
        urlretrieve(url, destination_path, reporthook)
        sys.stdout.write('\n')
    except (Exception, KeyboardInterrupt, SystemExit):
        print('Rolling back: remove partially downloaded file')
        os.remove(destination_path)
        raise


def unzip(compressed_path, output_dir):
    with zipfile.ZipFile(compressed_path, 'r') as f:
        f.extractall(output_dir)


def untar(compressed_path, output_dir):
    with tarfile.open(compressed_path) as f:
        f.extractall(output_dir)


def download_stanford_corenlp():
    url = 'http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip'
    temp_filepath = get_temp_filepath(create=True)
    download(url, temp_filepath)
    STANFORD_CORENLP_DIR.mkdir(parents=True, exist_ok=True)
    unzip(temp_filepath, STANFORD_CORENLP_DIR.parent)


def update_ucca_path():
    # HACK: Change vocab_path from relative to absolute path
    json_path = str(UCCA_PARSER_PATH) + '.nlp.json'
    with open(json_path, 'r') as f:
        config_json = json.load(f)
    config_json['vocab'] = str(UCCA_DIR / 'vocab/en_core_web_lg.csv')
    with open(json_path, 'w') as f:
        json.dump(config_json, f)


def download_ucca_model():
    url = 'https://github.com/huji-nlp/tupa/releases/download/v1.3.10/ucca-bilstm-1.3.10.tar.gz'
    temp_filepath = get_temp_filepath(create=True)
    download(url, temp_filepath)
    UCCA_DIR.mkdir(parents=True, exist_ok=True)
    untar(temp_filepath, UCCA_DIR)
    update_ucca_path()


def maybe_map_deprecated_test_set_to_new_test_set(test_set):
    '''Map deprecated test sets to new test sets'''
    deprecated_test_sets_map = {
        'turk': 'turkcorpus_test',
        'turk_valid': 'turkcorpus_valid',
    }
    if test_set in deprecated_test_sets_map:
        deprecated_test_set = test_set
        test_set = deprecated_test_sets_map[deprecated_test_set]
        warnings.warn(f'"{deprecated_test_set}" test set is deprecated. Please use "{test_set}" instead.')
    return test_set


def get_orig_sents(test_set):
    test_set = maybe_map_deprecated_test_set_to_new_test_set(test_set)
    return read_lines(TEST_SETS_PATHS[(test_set, 'orig')])


def get_refs_sents(test_set):
    test_set = maybe_map_deprecated_test_set_to_new_test_set(test_set)
    return [read_lines(ref_sents_path) for ref_sents_path in TEST_SETS_PATHS[(test_set, 'refs')]]


def get_system_outputs_dir(test_set):
    return SYSTEM_OUTPUTS_DIRS_MAP[test_set]
