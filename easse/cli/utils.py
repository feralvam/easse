import json
import os


def get_valid_test_sets(as_str=False):
    with open('easse/config.json', 'r') as config_file:
        config = json.load(config_file)

    if as_str:
        return ','.join(config["DATASETS"])
    else:
        return config["DATASETS"]


def get_valid_metrics(as_str=False):
    with open('easse/config.json', 'r') as config_file:
        config = json.load(config_file)

    if as_str:
        return ','.join(config["METRICS"])
    else:
        return config["METRICS"]


def read_file(filename):
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines
