from ucca import convert
from xml.etree.ElementTree import fromstring

import nltk


def get_scenes(P):
    """
    P is a ucca passage. Return all the scenes in each passage
    """
    scenes = [x for x in P.layer("1").all if x.tag == "FN" and x.is_scene()]
    y = P.layer("0")
    output = []
    for sc in scenes:
        p = [] 
        d = sc.get_terminals(False,True)
        for i in list(range(0, len(d))):
            p.append(d[i].position)
        output2 = []     
        for k in p:
            if(len(output2)) == 0:
               output2.append(str(y.by_position(k))) 
            elif str(y.by_position(k)) != output2[-1]:
               output2.append(str(y.by_position(k)))
            
        output.append(output2)        
    return output

    
def get_sentences(P):
    """
    P is the output of the simplification system. Return all the sentences in each passage
    """
    dirpath = '/Mypath/System_output'
    folder = nltk.data.find(dirpath)
    corpus_reader = nltk.corpus.PlaintextCorpusReader(folder, P)
    d = len(corpus_reader.sents())
    return corpus_reader.sents()[:d]


index = list(range(0, 100))


for t in index:
    f1 = open('UCCAannotated_source/%s.xml' %t)
    xml_string1 = f1.read()
    f1.close()
    xml_object1 = fromstring(xml_string1)
    P1 = convert.from_site(xml_object1)
    L1 = get_scenes(P1)
    L2 = get_sentences('%s.txt' %t)
    s = open('s%s.txt' %t, 'w')
    s.write('%s\n' %L1)
    s.write('%s\n' %L2)

    s.close()

