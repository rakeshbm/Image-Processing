#!/usr/bin/env python
# coding: utf-8




#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

def compute_IoU(b1,b2,iou_threshold=0.5):
    if b1[4]!=b2[4]:
        return False
    box1=b1[:4]
    box2=b2[:4]
    x1,y1,x2,y2=box1
    x_1,y_1,x_2,y_2=box2
    xA=max(x1,x_1)
    yA=max(y1,y_1)
    xB=min(x2,x_2)
    yB=min(y2,y_2)
    overlap_area=max(0,xB-xA+1)*max(0,yB-yA+1)
    union_area=max(0,y2-y1+1)*max(0,x2-x1+1)+max(0,y_2-y_1+1)*max(0,x_2-x_1+1)-overlap_area
    iou=overlap_area/union_area
    print ('IOU=',iou)
    return iou>=iou_threshold

def positiveEvaluation(gt_box, pr_box):
    m = len(gt_box)
    n = len(pr_box)
    class_score=[]
    if n==0:
        return []
    else:
        for b1 in pr_box:
            tmp=0
            for b2 in gt_box:
                if compute_IoU(b1,b2):
                    tmp+=1
                    class_score.append([b1[4],b1[5],1])
                    break
            if tmp==0:
                class_score.append([b1[4],b1[5],0])
    return class_score #[(class,confidence,'P/N')]

def compute_AP(class_score,c):

    class_score=sorted(class_score,key=lambda x:x[1],reverse=True)
    label,confidence,IsPositive=[x[0] for x in class_score],[x[1] for x in class_score],[x[2] for x in class_score]
    M=sum(IsPositive)#num of correct predictions
    N=len(label)
    print('class %d :number of correct predictions: %d,number of total predictions: %d'%(c,M,N))
    if M==0:
        print ('Average Precision of class %d : %f' %(c,0.0))
        return 0.0

    PR,Recall=[],[]
    #topN:top(n+1)
    for n in range(N):
        num_true=sum(IsPositive[:n+1])
        PR.append(num_true/(n+1))
        Recall.append(num_true/M)

    #Compute AP
    max_PR=[]#max precision for any recall>=recall_threshold （1/M, 2/M, ..., M/M）
    for j in range(M):
        recall_threshold=(j+1)/M
        ind=Recall.index(recall_threshold)
        max_PR.append(max(PR[ind:]))
    if len(max_PR)==0:
        AP= 0
    else:
        AP=sum(max_PR)/len(max_PR)
    print ('Average Precision of class %d : %f' %(c,AP))
    print('\n')
    return AP



def compute_mAP(class_score,class_idx):
    class_dict=[]
    print(class_idx)
    for c in class_idx:
        class_dict.append(filter(lambda x: x[0]==c, class_score))
    AP=[]
    for c in class_idx:
        ap=compute_AP(class_dict[c],c)
        AP.append(ap)

    if len(AP)==0:
        mAP=0
    else:
        mAP=sum(AP)/len(AP)
    print ('mean Average Precision: ',mAP)
    return mAP 







