import os
import random

trainval_percent = 0.0
train_percent = 0.0
xmlfilepath = 'testdata/annotation'
txtsavepath = 'testdata/Main'
total_xml = os.listdir(xmlfilepath)

num=len(total_xml)
list=range(num)
tv=int(num*trainval_percent)
tr=int(tv*train_percent)
trainval= random.sample(list,tv)
train=random.sample(trainval,tr)

ftest = open(txtsavepath+'/test.txt', 'w')

for i in list:
    name=total_xml[i][:-4]+'\n'
    if (not name =='0'+'\n'):
        print (name)
        if i not in trainval:
            ftest.write(name)


ftest.close()