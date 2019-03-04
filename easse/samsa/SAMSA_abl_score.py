from ucca import convert
from xml.etree.ElementTree import fromstring

import nltk
import ast


def get_num_scenes(P):
    """
    P is a ucca passage. Returns the number of scenes.
    """
    scenes = [x for x in P.layer("1").all if x.tag == "FN" and x.is_scene()]
    output = len(scenes)

    return output


def get_num_sentences(P):
    """
    P is the output of the simplification system. Return all the sentences in each passage
    """
    dirpath = '/Mypath/System_output' # Replace Zhu by Woodsend/Wubben/Narayan1/Narayan2/Narayan3/Simple for testing the different systems
    folder = nltk.data.find(dirpath)
    corpusReader = nltk.corpus.PlaintextCorpusReader(folder, P)

    return len(corpusReader.sents())


def get_cmrelations(P):
    """
    P is a ucca passage. Return all the most internal centers of main relations in each passage
    """
    scenes = [x for x in P.layer("1").all if x.tag == "FN" and x.is_scene()]
    m = []
    #c = []
    for sc in scenes:
        mrelations = [e.child for e in sc.outgoing if e.tag == 'P' or e.tag == 'S']
        for mr in mrelations:
            centers = [e.child for e in mr.outgoing if e.tag == 'C']
            if centers != []:
                while centers != []:
                    for c in centers:
                        ccenters = [e.child for e in c.outgoing if e.tag == 'C']
                    lcenters = centers
                    centers = ccenters
                m.append(lcenters)
            else:
                m.append(mrelations)

    y = P.layer("0")
    output = []
    for scp in m:
        for par in scp:
            output2 =[]
            p = []
            d = par.get_terminals(False,True)
            for i in list(range(0,len(d))):
                p.append(d[i].position)

            for k in p:

                if(len(output2)) == 0:
                    output2.append(str(y.by_position(k)))
                elif str(y.by_position(k)) != output2[-1]:
                    output2.append(str(y.by_position(k)))

        output.append(output2)

    return output


def get_cparticipants(P):
    """
    P is a ucca passage. Return all the minimal participant centers in each scene
    """
    scenes = [x for x in P.layer("1").all if x.tag == "FN" and x.is_scene()]
    n = []
    for sc in scenes:  #find participant nodes
        m = []
        participants = [e.child for e in sc.outgoing if e.tag == 'A']
        for pa in participants:
            centers = [e.child for e in pa.outgoing if e.tag == 'C' ]
            if centers != []:
                while centers != []:
                    for c in centers:
                        ccenters = [e.child for e in c.outgoing if e.tag == 'C' or e.tag == 'P' or e.tag == 'S']   # also addresses center Scenes
                    lcenters = centers
                    centers = ccenters
                m.append(lcenters)
            elif pa.is_scene():     #address the case of Participant Scenes
                scenters = [e.child for e in pa.outgoing if e.tag == 'P' or e.tag == 'S']
                for scc in scenters:
                    centers = [e.child for e in scc.outgoing if e.tag == 'C']
                    if centers != []:
                        while centers != []:
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
                        if centers != []:
                            while centers != []:
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
                par = [par[i:i+1] for i in range(0,len(par))]
                for c in par:
                    r.append(c)
                I.append([u,d])
            else:
                r.append(par)
        s.append(r)

    for scp in s:  # find the spans of the participant nodes
        output1 = []
        for [par] in scp:
            output2 =[]
            p = []
            d = par.get_terminals(False,True)
            for i in list(range(0,len(d))):
                p.append(d[i].position)

            for k in p:

                if(len(output2)) == 0:
                    output2.append(str(y.by_position(k)))
                elif str(y.by_position(k)) != output2[-1]:
                    output2.append(str(y.by_position(k)))
            output1.append(output2)
        output.append(output1)

    y = []   # unify spans in case of multiple centers
    for scp in output:
        x = []
        u = output.index(scp)
        for par in scp:
            for v in I:
                if par == output[v[0]][v[1]]:
                    for l in list(range(1,len(n[v[0]][v[1]]))):
                        par.append((output[v[0]][v[1]+l])[0])

                    x.append(par)
                elif all(par != output[v[0]][v[1]+l] for l in list(range(1,len(n[v[0]][v[1]])))):
                    x.append(par)
            if I == []:
                x.append(par)
        y.append(x)

    return y


index = list(range(0,100))

for t in index:
    f1 = open('UCCAannotated_source/%s.xml' %t)
    xml_string1 = f1.read()
    f1.close()
    xml_object1 = fromstring(xml_string1)
    P1 = convert.from_standard(xml_object1)  #from_site for semi-automatic SAMSA_abl
    L1 = get_num_scenes(P1)
    L2 = get_num_sentences('%s.txt' %t)
    M1 = get_cmrelations(P1)
    A1 = get_cparticipants(P1)

    if L1 < L2:
        score = 0

    elif L1 == L2:
        f1 = open('scene_sentence_alignment_output/a%s.txt' %t)   #Replace Zhu by Woodsend/Wubben/Narayan1/Narayan2/Narayan3/Simple for testing the different systems
        s = f1.read()
        f1.close()
        t = ast.literal_eval(s)
        match = []
        for i in list(range(0,L1)):
            match_value = 0
            for j in list(range(0,L2)):
                if len(t[i][j]) > match_value and j not in match:
                    match_value = len(t[i][j])
                    m = j
            match.append(m)
        scorem = []
        scorea = []
        for i in list(range(0,L1)):
            j = match[i]
            r = [t[i][j][k][0] for k in list(range(0,len(t[i][j])))]
            if M1[i]==[]:
               s = 0.5
            elif all(M1[i][l] in r for l in list(range(0,len(M1[i])))):
               s = 1
            else:
               s = 0
            scorem.append(s)
            sa = []
            if A1[i] == []:
                sa = [0.5]
                scorea.append(sa)
            else:
                for a in A1[i]:
                    if a == []:
                        p = 0.5
                    elif all(a[l] in r for l in list(range(0,len(a)))):
                        p = 1
                    else:
                        p = 0
                    sa.append(p)
                scorea.append(sa)

        scoresc = []
        for i in list(range(0,L1)):
            d = len(scorea[i])
            v = 0.5*scorem[i] + 0.5*(1/d)*sum(scorea[i])
            scoresc.append(v)
        score = (L2/(L1**2))*sum(scoresc)



    else:
        f1 = open('scene_sentence_alignment_output/a%s.txt' %t)
        s = f1.read()
        f1.close()
        t = ast.literal_eval(s)
        match = []
        for i in list(range(0,L1)):
            match_value = 0
            for j in list(range(0,L2)):
                if len(t[i][j]) > match_value:
                   match_value = len(t[i][j])
                   m = j
            match.append(m)
        scorem = []
        scorea = []
        for i in list(range(0,L1)):
            j = match[i]
            r = [t[i][j][k][0] for k in list(range(0,len(t[i][j])))]
            if M1[i]==[]:
                s = 0.5
            elif all(M1[i][l] in r for l in list(range(0,len(M1[i])))):
                s = 1
            else:
                s = 0
            scorem.append(s)
            sa = []
            if A1[i] == []:
                sa = [0.5]
                scorea.append(sa)
            else:
                for a in A1[i]:
                    if a == []:
                        p = 0.5
                    elif all(a[l] in r for l in list(range(0,len(a)))):
                        p = 1
                    else:
                        p = 0
                    sa.append(p)
                scorea.append(sa)

        scoresc = []
        for i in list(range(0,L1)):
            d = len(scorea[i])
            v = 0.5*scorem[i] + 0.5*(1/d)*sum(scorea[i])
            scoresc.append(v)
        score = (1/L1)*sum(scoresc)

    print(score)
