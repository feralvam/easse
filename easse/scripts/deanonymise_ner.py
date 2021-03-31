import argparse
import io
import sys
import torchfile


def open_file(file, mode='rt', encoding='utf-8'):
    return open(file, mode=mode, encoding=encoding, newline="\n")


def _replace_ner(sentence, ner_dict):
    """Replace the Named Entities in a sentence
    Args:
        sentence: str, sentences to be processed
        ner_dict: dictionary of {NER_tag: word} or an empty list
    Returns:
        processed sentence
    """
    if isinstance(ner_dict, (list, tuple)):
        # the map is empty, no NER in the sentence
        return sentence

    def replace_fn(token):
        # for compatability between python2 and 3
        # upper because the NER are upper-based
        if token.upper().encode() in ner_dict.keys():
            # lower case replaced words
            return ner_dict[token.upper().encode()].decode().lower()
        else:
            return token

    return " ".join(map(replace_fn, sentence.split()))


def _deanonymize_file(sys_output, ner_map_file, mode):
    # read in NER_Map
    ner_maps = torchfile.load(ner_map_file)
    # for compatibility between python2 and 3
    ner_map = ner_maps[mode.encode(encoding="utf-8")]

    # process sentences
    deanonymized_outputs = []
    if not len(sys_output) == len(ner_map):
        raise ValueError(f"sys_output and ner_map shape mismatch: {len(sys_output)} <> {len(ner_map)}")
    for raw_output, ner_dict in zip(sys_output, ner_map):
        deanonymized_output = _replace_ner(raw_output, ner_dict)
        deanonymized_outputs.append(deanonymized_output)

    return deanonymized_outputs


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="TODO.")
    arg_parser.add_argument('--input', '-i', type=str, default='-', help='Read input from a file instead of STDIN')
    arg_parser.add_argument('--mode', '-m', help="TODO.")
    arg_parser.add_argument('--ner_map_file', '-aner', help="TODO.")

    args = arg_parser.parse_args()

    inputfh = io.TextIOWrapper(sys.stdin.buffer) if args.input == '-' else open_file(args.input)
    sys_output = inputfh.readlines()
    sys_output = [d.strip() for d in sys_output]

    deanonymized_sys_outputs = _deanonymize_file(sys_output, mode=args.mode, ner_map_file=args.ner_map_file)

    print("\n".join(deanonymized_sys_outputs))
