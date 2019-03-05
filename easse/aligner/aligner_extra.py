import easse.aligner.aligner
import argparse
import easse.aligner.coreNlpUtil
import json
import os


def read_text_file(file_path):
    with open(file_path) as infile:
        return infile.read().splitlines()


def read_json_file(file_path):
    with open(file_path) as infile:
        return json.load(infile)['sentences']


def group_sentence_alignments(sent1_parse_lst, sent2_parse_lst, sent_aligns):
    sent1_parse = []
    sent2_parse = []
    sent1_map = {}
    sent2_map = {}
    for index_pair in sent_aligns:
        sent1_index, sent2_index = map(int, index_pair.strip().split('\t'))
        print sent1_index, sent2_index
        sent1_added = sent1_index in sent1_map
        sent2_added = sent2_index in sent2_map

        if sent1_added and not sent2_added:  # it is a split
            sent_group_index = sent1_map[sent1_index]
            if len(sent1_parse[sent_group_index]) > 1:  # check to prevent M-to-N alignments
                sent_group_index = len(sent1_parse)
                sent1_map[sent1_index] = sent_group_index
                sent2_map[sent2_index] = sent_group_index
                sent2_parse.insert(sent_group_index, [sent2_parse_lst[sent2_index]])
                sent1_parse.insert(sent_group_index, [sent1_parse_lst[sent1_index]])
            else:
                sent2_map[sent2_index] = sent_group_index
                sent2_parse[sent_group_index].append(sent2_parse_lst[sent2_index])
        elif sent2_added and not sent1_added:  # it is a join
            sent_group_index = sent2_map[sent2_index]
            if len(sent2_parse[sent_group_index]) > 1:  # check to prevent M-to-N alignments
                sent_group_index = len(sent2_parse)
                sent1_map[sent1_index] = sent_group_index
                sent2_map[sent2_index] = sent_group_index
                sent1_parse.insert(sent_group_index, [sent1_parse_lst[sent1_index]])
                sent2_parse.insert(sent_group_index, [sent2_parse_lst[sent2_index]])
            else:
                sent1_map[sent1_index] = sent_group_index
                sent1_parse[sent_group_index].append(sent1_parse_lst[sent1_index])
        else:
            # not sent1_added and not sent2_added: it is a new pair. We treat as 1-to-1
            # sent1_added and sent2_added: it is a M-to-N (not supported). We add it as a 1-to-1
            sent_group_index = len(sent1_parse)
            sent1_map[sent1_index] = sent_group_index
            sent2_map[sent2_index] = sent_group_index
            sent1_parse.insert(sent_group_index, [sent1_parse_lst[sent1_index]])
            sent2_parse.insert(sent_group_index, [sent2_parse_lst[sent2_index]])

    return zip(sent1_parse, sent2_parse)


if __name__ == '__main__':
    # create an Argument Parser to handle command line arguments
    parser = argparse.ArgumentParser(description="uses the monolingual-word-aligner on already parsed sentences")

    parser.add_argument('sent1parsepath', help="json file with parsed sentences to align from.")
    parser.add_argument('sent2parsepath', help="json file with parsed sentences to align to.")
    parser.add_argument('-sentalignspath', help="file with the sentence alignments. If not given, 1-to-1 is assumed.")
    parser.add_argument('-outputfilename', help="name of the file with the word alignments.", default='aligns.mwa')
    parser.add_argument('-outputfolder', help="folder where to put the file with the word alignments.", default="./")

    args = parser.parse_args()

    sent1_parse_lst = read_json_file(args.sent1parsepath)
    sent2_parse_lst = read_json_file(args.sent2parsepath)

    if args.sentalignspath is None:  # assume 1-to-1 alignment
        sent_aligns = ['{}\t{}'.format(i, i) for i in range(0, len(sent1_parse_lst))]
    else:    
        sent_aligns = read_text_file(args.sentalignspath)

    sents_info = group_sentence_alignments(sent1_parse_lst, sent2_parse_lst, sent_aligns)

    word_aligns = []
    for sent1_parse_json, sent2_parse_json in sents_info:
        sent1_parse_result = coreNlpUtil.format_json_parser_results(sent1_parse_json)
        sent2_parse_result = coreNlpUtil.format_json_parser_results(sent2_parse_json)
        # get the alignments (only indices)
        aligns, _ = aligner.align(sent1_parse_result, sent2_parse_result)
        # convert to pharaoh format: [[1, 1], [2, 2]] -> ['1-1', '2-2']
        aligns_pharaoh = ['-'.join([str(p[0]), str(p[1])]) for p in aligns]
        # create a single line to write: ['1-1', '2-2'] -> '1-1 2-2'
        aligns_line = ' '.join(aligns_pharaoh)
        word_aligns.append(aligns_line)
    
    aligns_file_path = os.path.join(args.outputfolder, args.outputfilename)
    with open(aligns_file_path, 'w') as aligns_file_path:
        aligns_file_path.write('\n'.join(word_aligns))
