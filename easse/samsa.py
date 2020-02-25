from typing import List
from ucca.core import Passage

from easse.utils.ucca_utils import get_scenes, ucca_parse_texts
from easse.aligner.aligner import align
from easse.aligner.corenlp_utils import syntactic_parse_texts
import easse.utils.preprocessing as utils_prep

from tqdm import tqdm


def align_scenes_sentences(scenes, synt_parse_sentences):
    # parse the scenes
    synt_parse_scenes = syntactic_parse_texts(scenes)
    all_scenes_alignments = []
    for synt_scene in synt_parse_scenes:
        scene_alignments = []
        for synt_sent in synt_parse_sentences:
            # word_alignments = [[word1_scene, word1_sentence], [word2_scene, word3_sentence], ...]
            word_alignments = align(synt_scene, synt_sent)[1]
            scene_alignments.append(word_alignments)
        all_scenes_alignments.append(scene_alignments)
    return all_scenes_alignments


def align_scenes_sentences_new(scenes, synt_parse_sentences, allow_mutiple_matches):
    synt_parse_scenes = syntactic_parse_texts(scenes)

    scenes_sents_aligns = []
    already_matched = []
    for synt_scene in synt_parse_scenes:
        max_sent_aligns = []
        for sent_num, synt_sent in enumerate(synt_parse_sentences):
            if not allow_mutiple_matches and sent_num in already_matched:
                continue
            word_alignments = align(synt_scene, synt_sent)[1]
            if len(word_alignments) > len(max_sent_aligns):
                max_sent_aligns = word_alignments
                max_sent_num = sent_num
        scenes_sents_aligns.append(max_sent_aligns)
        if len(max_sent_aligns) > 0:
            already_matched.append(max_sent_num)

    return scenes_sents_aligns


def get_num_scenes(ucca_passage: Passage):
    """
    Returns the number of scenes in the ucca_passage.
    """
    scenes = [x for x in ucca_passage.layer("1").all if x.tag == "FN" and x.is_scene()]
    return len(scenes)


def get_relations_minimal_centers(ucca_passage: Passage):
    """
    Return all the most internal centers of main relations in each passage
    """
    scenes = [x for x in ucca_passage.layer("1").all if x.tag == "FN" and x.is_scene()]
    minimal_centers = []
    for sc in scenes:
        min_relations = [e.child for e in sc.outgoing if e.tag == 'P' or e.tag == 'S']
        for mr in min_relations:
            centers = [e.child for e in mr.outgoing if e.tag == 'C']
            if centers:
                while centers:
                    for c in centers:
                        ccenters = [e.child for e in c.outgoing if e.tag == 'C']
                    lcenters = centers
                    centers = ccenters
                minimal_centers.append(lcenters)
            else:
                minimal_centers.append(min_relations)

    y = ucca_passage.layer("0")
    output = []
    for scp in minimal_centers:
        for par in scp:
            output2 = []
            positions = [d.position for d in par.get_terminals(False, True)]
            for pos in positions:
                if not output2:
                    output2.append(str(y.by_position(pos)))
                elif str(y.by_position(pos)) != output2[-1]:
                    output2.append(str(y.by_position(pos)))

        output.append(output2)

    return output


def get_participants_minimal_centers(P: Passage):
    """
    P is a ucca passage. Return all the minimal participant centers in each scene
    """
    scenes = [x for x in P.layer("1").all if x.tag == "FN" and x.is_scene()]
    n = []
    for sc in scenes:  # find participant nodes
        m = []
        participants = [e.child for e in sc.outgoing if e.tag == 'A']
        for pa in participants:
            centers = [e.child for e in pa.outgoing if e.tag == 'C']
            if centers:
                while centers:
                    for c in centers:
                        ccenters = [e.child for e in c.outgoing if e.tag == 'C' or e.tag == 'P' or e.tag == 'S']   #also addresses center Scenes
                    lcenters = centers
                    centers = ccenters
                m.append(lcenters)
            elif pa.is_scene():  # address the case of Participant Scenes
                scenters = [e.child for e in pa.outgoing if e.tag == 'P' or e.tag == 'S']
                for scc in scenters:
                    centers = [e.child for e in scc.outgoing if e.tag == 'C']
                    if centers:
                        while centers:
                            for c in centers:
                                ccenters = [e.child for e in c.outgoing if e.tag == 'C']
                            lcenters = centers
                            centers = ccenters
                        m.append(lcenters)
                    else:
                        m.append(scenters)
            elif any(e.tag == "H" for e in pa.outgoing):  # address the case of multiple parallel Scenes inside a participant
                hscenes = [e.child for e in pa.outgoing if e.tag == 'H']
                mh = []
                for h in hscenes:
                    hrelations = [e.child for e in h.outgoing if e.tag == 'P' or e.tag == 'S']  # in case of multiple parallel scenes we generate new multiple centers
                    for hr in hrelations:
                        centers = [e.child for e in hr.outgoing if e.tag == 'C']
                        if centers:
                            while centers:
                                for c in centers:
                                    ccenters = [e.child for e in c.outgoing if e.tag == 'C']
                                lcenters = centers
                                centers = ccenters
                            mh.append(lcenters[0])
                        else:
                            mh.append(hrelations[0])
                m.append(mh)
            else:
                m.append([pa])

        n.append(m)

    y = P.layer("0")  # find cases of multiple centers
    output = []
    s = []
    I = []
    for scp in n:
        r = []
        u = n.index(scp)
        for par in scp:
            if len(par) > 1:
                d = scp.index(par)
                par = [par[i:i+1] for i in range(len(par))]
                for c in par:
                    r.append(c)
                I.append([u,d])
            else:
                r.append(par)
        s.append(r)

    for scp in s:  # find the spans of the participant nodes
        output1 = []
        for [par] in scp:
            output2 = []
            p = []
            d = par.get_terminals(False,True)
            for i in range(0, len(d)):
                p.append(d[i].position)

            for k in p:
                if len(output2) == 0:
                    output2.append(str(y.by_position(k)))
                elif str(y.by_position(k)) != output2[-1]:
                    output2.append(str(y.by_position(k)))
            output1.append(output2)
        output.append(output1)

    y = []  # unify spans in case of multiple centers
    for scp in output:
        x = []
        u = output.index(scp)
        for par in scp:
            for v in I:
                if par == output[v[0]][v[1]]:
                    for l in range(1,len(n[v[0]][v[1]])):
                        par.append((output[v[0]][v[1]+l])[0])

                    x.append(par)
                elif all(par != output[v[0]][v[1]+l] for l in range(1, len(n[v[0]][v[1]]))):
                    x.append(par)
            if not I:
                x.append(par)
        y.append(x)

    return y


def compute_samsa(orig_ucca_passage: Passage, sys_synt_parse):
    orig_scenes = get_scenes(orig_ucca_passage)

    orig_num_scenes = len(orig_scenes)
    sys_num_sents = len(sys_synt_parse)

    if orig_num_scenes < sys_num_sents:
        score = 0.0
    else:
        rel_min_centers = get_relations_minimal_centers(orig_ucca_passage)
        part_min_centers = get_participants_minimal_centers(orig_ucca_passage)

        allow_mutiple_matches = orig_num_scenes > sys_num_sents

        orig_scenes_sys_sentences_alignments = align_scenes_sentences_new(orig_scenes, sys_synt_parse,
                                                                          allow_mutiple_matches)
        scorem = []
        scorea = []
        for scene_index, scene_sent_word_aligns in enumerate(orig_scenes_sys_sentences_alignments):
            r = [scene_word for scene_word, _ in scene_sent_word_aligns]
            if not rel_min_centers[scene_index]:
                s = 0.5
            elif all(rel_min_centers[scene_index][l] in r for l in range(len(rel_min_centers[scene_index]))):
                s = 1.0
            else:
                s = 0.0
            scorem.append(s)

            sa = []
            if not part_min_centers[scene_index]:
                sa = [0.5]
                scorea.append(sa)
            else:
                for a in part_min_centers[scene_index]:
                    if not a:
                        p = 0.5
                    elif all(a[l] in r for l in range(len(a))):
                        p = 1
                    else:
                        p = 0
                    sa.append(p)
                scorea.append(sa)

        scoresc = []
        for i in range(orig_num_scenes):
            d = len(scorea[i])
            v = 0.5*scorem[i] + 0.5*(1/d)*sum(scorea[i])
            scoresc.append(v)
        score = (sys_num_sents/(orig_num_scenes**2))*sum(scoresc)
    return score


def corpus_samsa(orig_sentences: List[str], sys_outputs: List[str], lowercase: bool = False, tokenizer: str = '13a',
                 verbose: bool = False):
    print('Warning: SAMSA metric is long to compute, disable it if you need fast evaluation.')
    orig_sentences = [utils_prep.normalize(sent, lowercase, tokenizer) for sent in orig_sentences]
    orig_ucca_sents = ucca_parse_texts(orig_sentences)

    sys_outputs = [utils_prep.normalize(output, lowercase, tokenizer) for output in sys_outputs]
    sys_synt_outputs = syntactic_parse_texts(sys_outputs, tokenize=False, sentence_split=True, verbose=verbose)

    if verbose:
        print("Computing SAMSA score...")

    samsa_score = 0.0
    for orig_ucca, sys_synt in tqdm(zip(orig_ucca_sents, sys_synt_outputs), disable=(not verbose)):
        samsa_score += compute_samsa(orig_ucca, sys_synt)

    samsa_score /= len(orig_sentences)

    return 100. * samsa_score
