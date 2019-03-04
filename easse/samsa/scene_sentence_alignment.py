from aligner import *
import ast

index = list(range(0,100))

for t in index:
    f1 = open('scene_sentence_extraction_output/s%s.txt' %t)
    lines = f1.readlines()
    f1.close()
    l1 = lines[0]
    l2 = lines[1]
    vl1 = ast.literal_eval(l1)
    vl2 = ast.literal_eval(l2)
    output = []
    for i in list(range(0,len(vl1))):
        output1 = []
        for j in list(range(0,len(vl2))):
            a = align(vl1[i],vl2[j])
            output1.append(a[1])
        output.append(output1)
                      
    s = open('a%s.txt' %t, 'w')
    s.write(str(output))
    s.close()

 



