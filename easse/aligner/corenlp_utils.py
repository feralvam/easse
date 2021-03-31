from typing import List
import os

from stanfordnlp.server import CoreNLPClient
from tqdm import tqdm

from easse.utils.resources import download_stanford_corenlp
from easse.utils.constants import STANFORD_CORENLP_DIR


def _format_token_info(sent_json):
    token_lst = sent_json["tokens"]
    tokens = []
    sent_str = ""
    for token in token_lst:
        tokens.append(token["word"])
        sent_str += (
            f"[Text={token['originalText']} CharacterOffsetBegin={token['characterOffsetBegin']} "
            f"CharacterOffsetEnd={token['characterOffsetEnd']} PartOfSpeech={token['pos']} "
            f"Lemma={token['lemma']} NamedEntityTag={token['ner']}] "
        )

    return tokens, sent_str


def _get_depnode_index(node_id, dep_parse):
    index = 0
    for dep_node in dep_parse:
        if dep_node["governor"] == node_id:
            return index
        index += 1


def _get_depnode_index_by_label(node_deplabel, dep_parse, node_ids):
    index = 0
    for dep_node in dep_parse:
        if dep_node["dep"] == node_deplabel and dep_node["governor"] in node_ids:
            return index
        index += 1


def _collapse_dependencies(dependency_parse):
    dep_tree_formatted = []
    for dep_node in dependency_parse:
        dep_rel = dep_node["dep"].lower()
        dependent_gloss = dep_node["dependentGloss"]
        dependent = dep_node["dependent"]

        if dep_rel == "prep":
            aux_dep_node_index = _get_depnode_index(dep_node["dependent"], dependency_parse)
            if aux_dep_node_index:
                dep_rel += f"_{dep_node['dependentGloss']}"
                aux_dep_node = dependency_parse[aux_dep_node_index]
                dependent_gloss = aux_dep_node["dependentGloss"]
                dependent = aux_dep_node["dependent"]
        elif dep_rel == "conj":
            aux_dep_node_index = _get_depnode_index_by_label(
                "cc",
                dependency_parse,
                [dep_node["dependent"], dep_node["governor"]],
            )
            if aux_dep_node_index:
                aux_dep_node = dependency_parse[aux_dep_node_index]
                dep_rel += f"_{aux_dep_node['dependentGloss']}"
            else:
                continue
        elif dep_rel in ["cc", "pobj"]:
            continue

        dep_tree_formatted.append(
            [
                dep_rel,
                f"{dep_node['governorGloss']}-{dep_node['governor']}",
                f"{dependent_gloss}-{dependent}",
            ]
        )

    return dep_tree_formatted


def format_parser_output(sentence_parse):
    results = {"sentences": []}
    for sent_json in sentence_parse:
        sent_formatted = {"words": []}
        # format the information of each word
        for token in sent_json["tokens"]:
            word = token["originalText"]
            attributes = {
                "CharacterOffsetBegin": str(token["characterOffsetBegin"]),
                "CharacterOffsetEnd": str(token["characterOffsetEnd"]),
                "PartOfSpeech": token["pos"],
                "Lemma": token["lemma"],
                "NamedEntityTag": token["ner"],
            }
            sent_formatted["words"].append((word, attributes))

        sent_formatted["text"] = " ".join([word for word, _ in sent_formatted["words"]])
        sent_formatted["dependencies"] = _collapse_dependencies(sent_json["basicDependencies"])
        if "parse" in sent_json:
            sent_formatted["parse"] = sent_json["parse"]

        results["sentences"].append(sent_formatted)

    return results


def syntactic_parse_texts(
    texts: List[str],
    tokenize=False,
    sentence_split=False,
    verbose=False,
    with_constituency_parse=False,
):
    corenlp_annotators = [
        "tokenize",
        "ssplit",
        "pos",
        "lemma",
        "ner",
        "depparse",
    ]
    if with_constituency_parse:
        corenlp_annotators.append("parse")
    annotators_properties = {
        "tokenize.whitespace": not tokenize,
        "ssplit.eolonly": not sentence_split,
        "depparse.model": "edu/stanford/nlp/models/parser/nndep/english_SD.gz",
        "outputFormat": "json",
    }
    if not STANFORD_CORENLP_DIR.exists():
        download_stanford_corenlp()
    os.environ["CORENLP_HOME"] = str(STANFORD_CORENLP_DIR)

    parse_results = []

    with CoreNLPClient(
        annotators=corenlp_annotators,
        properties=annotators_properties,
        threads=40,
    ) as client:
        for text in tqdm(texts, disable=(not verbose)):
            if isinstance(text, List):
                text = " ".join(text)
            raw_parse_result = client.annotate(text)
            parse_result = format_parser_output(raw_parse_result["sentences"])

            if len(parse_result["sentences"]) > 1 and not sentence_split:
                parse_result = join_parse_result(parse_result)
            elif sentence_split:
                parse_result = split_parse_result(parse_result["sentences"])

            parse_results.append(parse_result)

    return parse_results


def split_parse_result(parse_result):
    split_results = []
    for sent_json in parse_result:
        split_results.append({"sentences": [sent_json]})
    return split_results


def join_parse_result(parseResult):
    wordOffset = 0

    for i in range(len(parseResult["sentences"])):

        if i > 0:
            for j in range(len(parseResult["sentences"][i]["dependencies"])):

                for k in range(1, 3):
                    tokens = parseResult["sentences"][i]["dependencies"][j][k].split("-")
                    if tokens[0] == "ROOT":
                        newWordIndex = 0
                    else:
                        if not tokens[
                            len(tokens) - 1
                        ].isdigit():  # forced to do this because of entries like u"lost-8'" in parseResult
                            continue
                        newWordIndex = int(tokens[len(tokens) - 1]) + wordOffset
                    if len(tokens) == 2:
                        parseResult["sentences"][i]["dependencies"][j][k] = tokens[0] + "-" + str(newWordIndex)
                    else:
                        w = ""
                        for l in range(len(tokens) - 1):
                            w += tokens[l]
                            if l < len(tokens) - 2:
                                w += "-"
                        parseResult["sentences"][i]["dependencies"][j][k] = w + "-" + str(newWordIndex)

        wordOffset += len(parseResult["sentences"][i]["words"])

    # merge information of all sentences into one
    for i in range(1, len(parseResult["sentences"])):
        parseResult["sentences"][0]["text"] += " " + parseResult["sentences"][i]["text"]
        for jtem in parseResult["sentences"][i]["dependencies"]:
            parseResult["sentences"][0]["dependencies"].append(jtem)
        for jtem in parseResult["sentences"][i]["words"]:
            parseResult["sentences"][0]["words"].append(jtem)

    # remove all but the first entry
    parseResult["sentences"] = parseResult["sentences"][0:1]

    return parseResult


def nerWordAnnotator(parseResult):
    res = []

    wordIndex = 1
    for i in range(len(parseResult["sentences"][0]["words"])):
        tag = [
            [
                parseResult["sentences"][0]["words"][i][1]["CharacterOffsetBegin"],
                parseResult["sentences"][0]["words"][i][1]["CharacterOffsetEnd"],
            ],
            wordIndex,
            parseResult["sentences"][0]["words"][i][0],
            parseResult["sentences"][0]["words"][i][1]["NamedEntityTag"],
        ]
        wordIndex += 1

        if tag[3] != "O":
            res.append(tag)

    return res


def ner(parseResult):

    nerWordAnnotations = nerWordAnnotator(parseResult)

    namedEntities = []
    currentNE = []
    currentCharacterOffsets = []
    currentWordOffsets = []

    for i in range(len(nerWordAnnotations)):

        if i == 0:
            currentNE.append(nerWordAnnotations[i][2])
            currentCharacterOffsets.append(nerWordAnnotations[i][0])
            currentWordOffsets.append(nerWordAnnotations[i][1])
            if len(nerWordAnnotations) == 1:
                namedEntities.append(
                    [
                        currentCharacterOffsets,
                        currentWordOffsets,
                        currentNE,
                        nerWordAnnotations[i - 1][3],
                    ]
                )
                break
            continue

        if (
            nerWordAnnotations[i][3] == nerWordAnnotations[i - 1][3]
            and nerWordAnnotations[i][1] == nerWordAnnotations[i - 1][1] + 1
        ):
            currentNE.append(nerWordAnnotations[i][2])
            currentCharacterOffsets.append(nerWordAnnotations[i][0])
            currentWordOffsets.append(nerWordAnnotations[i][1])
            if i == len(nerWordAnnotations) - 1:
                namedEntities.append(
                    [
                        currentCharacterOffsets,
                        currentWordOffsets,
                        currentNE,
                        nerWordAnnotations[i][3],
                    ]
                )
        else:
            namedEntities.append(
                [
                    currentCharacterOffsets,
                    currentWordOffsets,
                    currentNE,
                    nerWordAnnotations[i - 1][3],
                ]
            )
            currentNE = [nerWordAnnotations[i][2]]
            currentCharacterOffsets = []
            currentCharacterOffsets.append(nerWordAnnotations[i][0])
            currentWordOffsets = []
            currentWordOffsets.append(nerWordAnnotations[i][1])
            if i == len(nerWordAnnotations) - 1:
                namedEntities.append(
                    [
                        currentCharacterOffsets,
                        currentWordOffsets,
                        currentNE,
                        nerWordAnnotations[i][3],
                    ]
                )

    return namedEntities


def posTag(parseResult):

    res = []

    wordIndex = 1
    for i in range(len(parseResult["sentences"][0]["words"])):
        tag = [
            [
                parseResult["sentences"][0]["words"][i][1]["CharacterOffsetBegin"],
                parseResult["sentences"][0]["words"][i][1]["CharacterOffsetEnd"],
            ],
            wordIndex,
            parseResult["sentences"][0]["words"][i][0],
            parseResult["sentences"][0]["words"][i][1]["PartOfSpeech"],
        ]
        wordIndex += 1
        res.append(tag)

    return res


def lemmatize(parseResult):
    res = []

    wordIndex = 1
    for i in range(len(parseResult["sentences"][0]["words"])):
        tag = [
            [
                parseResult["sentences"][0]["words"][i][1]["CharacterOffsetBegin"],
                parseResult["sentences"][0]["words"][i][1]["CharacterOffsetEnd"],
            ],
            wordIndex,
            parseResult["sentences"][0]["words"][i][0],
            parseResult["sentences"][0]["words"][i][1]["Lemma"],
        ]
        wordIndex += 1
        res.append(tag)
    return res


def dependencyParseAndPutOffsets(parseResult):
    # returns dependency parse of the sentence where each item is of the form:
    # (rel, left{charStartOffset, charEndOffset, wordNumber}, right{charStartOffset, charEndOffset, wordNumber})

    dParse = parseResult["sentences"][0]["dependencies"]
    words = parseResult["sentences"][0]["words"]

    result = []

    for item in dParse:
        # copy 'rel'
        newItem = [item[0]]

        # construct and append entry for 'left'
        left = item[1][0 : item[1].rindex("-")]
        wordNumber = item[1][item[1].rindex("-") + 1 :]
        if not wordNumber.isdigit():
            continue

        # left += (f"{words[int(wordNumber) - 1][1]['CharacterOffsetBegin']} "
        #          f"{words[int(wordNumber) - 1][1]['CharacterOffsetEnd']} "
        #          f"{wordNumber}")

        left += (
            "{"
            + words[int(wordNumber) - 1][1]["CharacterOffsetBegin"]
            + " "
            + words[int(wordNumber) - 1][1]["CharacterOffsetEnd"]
            + " "
            + wordNumber
            + "}"
        )
        newItem.append(left)

        # construct and append entry for 'right'
        right = item[2][0 : item[2].rindex("-")]
        wordNumber = item[2][item[2].rindex("-") + 1 :]
        if not wordNumber.isdigit():
            continue
        right += (
            "{"
            + words[int(wordNumber) - 1][1]["CharacterOffsetBegin"]
            + " "
            + words[int(wordNumber) - 1][1]["CharacterOffsetEnd"]
            + " "
            + wordNumber
            + "}"
        )
        newItem.append(right)

        result.append(newItem)

    return result


def findParents(dependencyParse, wordIndex, word):
    # word index assumed to be starting at 1
    # the third parameter is needed because of the collapsed representation of the dependencies...

    wordsWithIndices = (
        (
            int(item[2].split("{")[1].split("}")[0].split(" ")[2]),
            item[2].split("{")[0],
        )
        for item in dependencyParse
    )
    wordsWithIndices = list(set(wordsWithIndices))
    wordsWithIndices = sorted(wordsWithIndices, key=lambda item: item[0])

    wordIndexPresentInTheList = False
    for item in wordsWithIndices:
        if item[0] == wordIndex:
            wordIndexPresentInTheList = True
            break

    parentsWithRelation = []

    if wordIndexPresentInTheList:
        for item in dependencyParse:
            currentIndex = int(item[2].split("{")[1].split("}")[0].split(" ")[2])
            if currentIndex == wordIndex:
                parentsWithRelation.append(
                    [
                        int(item[1].split("{")[1].split("}")[0].split(" ")[2]),
                        item[1].split("{")[0],
                        item[0],
                    ]
                )
    else:
        # find the closest following word index which is in the list
        nextIndex = 0
        for i in range(len(wordsWithIndices)):
            if wordsWithIndices[i][0] > wordIndex:
                nextIndex = wordsWithIndices[i][0]
                break
        if nextIndex == 0:
            return []  # ?
        for i in range(len(dependencyParse)):
            if int(dependencyParse[i][2].split("{")[1].split("}")[0].split(" ")[2]) == nextIndex:
                pos = i
                break
        for i in range(pos, len(dependencyParse)):
            if "_" in dependencyParse[i][0] and word in dependencyParse[i][0]:
                parent = [
                    int(dependencyParse[i][1].split("{")[1].split("}")[0].split(" ")[2]),
                    dependencyParse[i][1].split("{")[0],
                    dependencyParse[i][0],
                ]
                parentsWithRelation.append(parent)
                break

    return parentsWithRelation


def findChildren(dependencyParse, wordIndex, word):
    # word index assumed to be starting at 1
    # the third parameter is needed because of the collapsed representation of the dependencies...

    wordsWithIndices = (
        (
            int(item[2].split("{")[1].split("}")[0].split(" ")[2]),
            item[2].split("{")[0],
        )
        for item in dependencyParse
    )
    wordsWithIndices = list(set(wordsWithIndices))
    wordsWithIndices = sorted(wordsWithIndices, key=lambda item: item[0])

    wordIndexPresentInTheList = False
    for item in wordsWithIndices:
        if item[0] == wordIndex:
            wordIndexPresentInTheList = True
            break

    childrenWithRelation = []

    if wordIndexPresentInTheList:
        for item in dependencyParse:
            currentIndex = int(item[1].split("{")[1].split("}")[0].split(" ")[2])
            if currentIndex == wordIndex:
                childrenWithRelation.append(
                    [
                        int(item[2].split("{")[1].split("}")[0].split(" ")[2]),
                        item[2].split("{")[0],
                        item[0],
                    ]
                )
    else:
        # find the closest following word index which is in the list
        nextIndex = 0
        for i in range(len(wordsWithIndices)):
            if wordsWithIndices[i][0] > wordIndex:
                nextIndex = wordsWithIndices[i][0]
                break

        if nextIndex == 0:
            return []
        for i in range(len(dependencyParse)):
            if int(dependencyParse[i][2].split("{")[1].split("}")[0].split(" ")[2]) == nextIndex:
                pos = i
                break
        for i in range(pos, len(dependencyParse)):
            if "_" in dependencyParse[i][0] and word in dependencyParse[i][0]:
                child = [
                    int(dependencyParse[i][2].split("{")[1].split("}")[0].split(" ")[2]),
                    dependencyParse[i][2].split("{")[0],
                    dependencyParse[i][0],
                ]
                childrenWithRelation.append(child)
                break

    return childrenWithRelation
