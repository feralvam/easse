from easse.aligner import align


def align_scenes_sentences(scenes, sentences):
    all_scenes_alignments = []
    for scene in scenes:
        scene_alignments = []
        for sentence in sentences:
            # word_alignments = [[word1_scene, word1_sentence], [word2_scene, word3_sentence], ...]
            word_alignments = align(scene, sentence)[1]
            scene_alignments.append(word_alignments)
        all_scenes_alignments.append(scene_alignments)
    return all_scenes_alignments
