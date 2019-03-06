def _format_token_info(sent_json):
    token_lst = sent_json['tokens']
    tokens = []
    sent_str = ""
    for token in token_lst:
        tokens.append(token['word'])
        sent_str += (f"[Text={token['originalText']} CharacterOffsetBegin={token['characterOffsetBegin']} "
                     f"CharacterOffsetEnd={token['characterOffsetEnd']} PartOfSpeech={token['pos']} "
                     f"Lemma={token['lemma']} NamedEntityTag={token['ner']}] ")

    return tokens, sent_str


def _get_depnode_index(node_id, dep_parse):
    index = 0
    for dep_node in dep_parse:
        if dep_node['governor'] == node_id:
            return index
        index += 1


def _get_depnode_index_by_label(node_deplabel, dep_parse, node_ids):
    index = 0
    for dep_node in dep_parse:
        if dep_node['dep'] == node_deplabel and dep_node['governor'] in node_ids:
            return index
        index += 1


def format_dependency_parse_tree(dependency_parse):
    dep_tree_formatted = []
    for dep_node in dependency_parse:
        dep_rel = dep_node['dep'].lower()
        dependent_gloss = dep_node['dependentGloss']
        dependent = dep_node['dependent']

        if dep_rel == 'prep':
            aux_dep_node_index = _get_depnode_index(dep_node['dependent'], dependency_parse)
            if aux_dep_node_index:
                dep_rel += f"_{dep_node['dependentGloss']}"
                aux_dep_node = dependency_parse[aux_dep_node_index]
                dependent_gloss = aux_dep_node['dependentGloss']
                dependent = aux_dep_node['dependent']
        elif dep_rel == 'conj':
            aux_dep_node_index = _get_depnode_index_by_label('cc', dependency_parse,
                                                             [dep_node['dependent'], dep_node['governor']])
            if aux_dep_node_index:
                aux_dep_node = dependency_parse[aux_dep_node_index]
                dep_rel += f"_{aux_dep_node['dependentGloss']}"
            else:
                continue
        elif dep_rel in ['cc', 'pobj']:
            continue

        dep_tree_formatted.append([dep_rel,
                                   f"{dep_node['governorGloss']}-{dep_node['governor']}",
                                   f"{dependent_gloss}-{dependent}"])

    return dep_tree_formatted
