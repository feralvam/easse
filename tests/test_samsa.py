import pytest
import csv

from easse.samsa.ucca_utils import get_scenes, ucca_parse_texts
from easse import samsa


# def test_get_scenes():
#     text = 'You are waiting for a train , this train will take you far away .'
#     expected_scenes = [
#         ['You', 'are', 'waiting', 'for', 'a', 'train'],
#         ['this', 'train', 'will', 'take', 'you', 'far', 'away']
#     ]
#     scenes = get_scenes(ucca_parse_text(text))
#     assert scenes == expected_scenes


def test_samsa_score_sentence():
    # orig_sentence = "You are waiting for a train , this train will take you far away ."
    # sys_output = "You are waiting for a train . A train that will take you far away ."
    # samsa_score = samsa.corpus_samsa([orig_sentence], [sys_output], lowercase=True)
    # assert samsa_score == pytest.approx(1.0)

    # orig_sentence = ("The river is indeed an ever-present part of the city's decor , and the official entrance "
    #                  "to Lisbon is a broad marble stair mounting from the water to the vast , "
    #                  "arcaded Commerce Square ( Praca do Comercio) .")
    # sys_output = ("The river is indeed an ever-present part of the city's decor , and the entrance to Lisbon "
    #               "is a broad marble stair mounting from the water to the covering , arcaded Commerce "
    #               "Square ( Praca do Comercio) .")
    # samsa_score = samsa.corpus_samsa([orig_sentence], [sys_output], lowercase=True)
    # assert samsa_score == pytest.approx(0.25)

    # orig_sentence = ("The second largest city of Russia and one of the world's major cities , "
    #                  "St . Petersburg has played a vital role in Russian history .")
    # sys_output = ("The second largest city of Russia and one of the world's major cities , "
    #               "St Petersburg , and has played a vital role in Russian history .")
    # samsa_score = samsa.corpus_samsa([orig_sentence], [sys_output], lowercase=True)
    # assert samsa_score == pytest.approx(0.833333333)

    # orig_sentence = ("The incident followed the killing in August of five Egyptian security guards by "
    #                  "Israeli soldiers pursuing militants who had ambushed and killed eight Israelis "
    #                  "along the Israeli-Egyptian border.")
    # sys_output = ("The incident followed the killing in August by Israeli soldiers. "
    #               "Israeli soldiers pursued militants. "
    #               "Militants had ambushed and killed eight Israelis along the Israeli-Egyptian border.")
    # samsa_score = samsa.corpus_samsa([orig_sentence], [sys_output], lowercase=True)
    # assert samsa_score == pytest.approx(0.71875)

    # orig_sentence = ("The injured man was able to drive his car to Cloverhill Prison where he got help. "
    #                  "He is being treated at Tallaght  Hospital but his injuries are not thought to be life-threatening.")
    # sys_output = "The injured man drive his car to Cloverhill Prison he got help."
    # samsa_score = samsa.corpus_samsa([orig_sentence], [sys_output])
    # assert samsa_score == pytest.approx(0.2222222222222222)

    orig_sentence = ("for example , king bhumibol was born on monday , "
                     "so on his birthday throughout thailand will be decorated with yellow color .")
    sys_output = ("for example , king bhumibol was born on monday , "
                  "so on his birthday throughout thailand will be decorated with yellow color .")
    samsa_score = samsa.corpus_samsa([orig_sentence], [sys_output], lowercase=False)
    print(samsa_score)


# def test_samsa_score_qats():
#     # read the sentence pairs from QATS test set
#     with open("data/qats/test.shared-task.tsv", newline='') as qats_file:
#         reader = csv.reader(qats_file, delimiter='\t')
#         next(reader, None)  # skip the header
#         orig_sentences = []
#         sys_outputs = []
#         for row in reader:
#             orig_sentences.append(row[0])
#             sys_outputs.append(row[1])
#
#     samsa_score = samsa.corpus_samsa(orig_sentences, sys_outputs, lowercase=True)
#     print(samsa_score)
