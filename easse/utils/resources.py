import json
import os
import sys
import tarfile
import time
from urllib.request import urlretrieve
import zipfile

from easse.utils.constants import DATA_DIR, STANFORD_CORENLP_DIR, UCCA_DIR, UCCA_PARSER_PATH
from easse.utils.helpers import get_temp_filepath, read_lines


def reporthook(count, block_size, total_size):
    # Download progress bar
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size_mb = count * block_size / (1024 * 1024)
    speed = progress_size_mb / duration
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
    temp_filepath = get_temp_filepath()
    download(url, temp_filepath)
    STANFORD_CORENLP_DIR.mkdir(parents=True, exist_ok=True)
    unzip(temp_filepath, STANFORD_CORENLP_DIR.parent)


def download_ucca_model():
    url = 'https://github.com/huji-nlp/tupa/releases/download/v1.3.10/ucca-bilstm-1.3.10.tar.gz'
    temp_filepath = get_temp_filepath()
    download(url, temp_filepath)
    UCCA_DIR.mkdir(parents=True, exist_ok=True)
    untar(temp_filepath, UCCA_DIR)
    # HACK: Change vocab_path from relative to absolute path
    json_path = str(UCCA_PARSER_PATH) + '.nlp.json'
    with open(json_path, 'r') as f:
        config_json = json.load(f)
    config_json['vocab'] = str(UCCA_DIR / config_json['vocab'])
    with open(json_path, 'w') as f:
            json.dump(config_json, f)


def get_turk_orig_sents(phase):
    assert phase in ['valid', 'test']
    if phase == 'valid':
        phase = 'tune'
    return read_lines(DATA_DIR / f'test_sets/turk/{phase}.8turkers.tok.norm')


def get_turk_refs_sents(phase):
    assert phase in ['valid', 'test']
    if phase == 'valid':
        phase = 'tune'
    return [read_lines(DATA_DIR / f'test_sets/turk/{phase}.8turkers.tok.turk.{i}')
            for i in range(8)]


def get_pwkp_orig_sents(phase):
    assert phase in ['valid', 'test']
    return read_lines(DATA_DIR / f'test_sets/pwkp/pwkp.{phase}.src')


def get_pwkp_refs_sents(phase):
    assert phase in ['valid', 'test']
    return [read_lines(DATA_DIR / f'test_sets/pwkp/pwkp.{phase}.dst')]


def get_hsplit_orig_sents():
    return get_turk_orig_sents(phase='test')[:70]


def get_hsplit_refs_sents():
    return [read_lines(DATA_DIR / f'test_sets/hsplit/hsplit.tok.{i+1}')
            for i in range(4)]
