# -*- coding: cp936 -*-
from math import log
import operator
#������Ϣ��
#���������ݼ�
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel]=0;
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

#�������ݼ�
def createDataSet():
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return  dataSet , labels

#���ո�����������������
#�������������ֵ����ݼ����������ݼ�����������Ҫ���ص�������ֵ
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:]) #a.extend(b)�õ�һ������a,b����Ԫ�ص��б�
            retDataSet.append(reduceFeatVec)#a.append(b)�б�õ����ĸ�Ԫ��Ϊb
    return  retDataSet

#ѡ����õ����ݼ����ַ�ʽ(����Ϣ����Ϊ��׼)
#���������ݼ�
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1   #��������
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):   #����������i��
        featureSet = set([example[i] for example in dataSet])   #��i������ȡֵ����
        newEntropy= 0.0
        for value in featureSet:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)   #��������������Ӧ��entropy
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#ͶƱ���������һ��
#������������б�
def majorityCnt(classList):
    classCount ={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#����������
#������ѵ���������Ա�ǩ
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):    #�����ȫ��ͬ��ֹͣ��������  �������ǩ-Ҷ�ӽڵ�
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)       #���������е�����ʱ���س��ִ�������
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    #del(labels[bestFeat])
    labelstmp = [] #ȥ�����ŵ�����
    for feat in labels:
        if feat != labels[bestFeat]:
            labelstmp.append(feat)
    featValues = [example[bestFeat] for example in dataSet]    #�õ����б�������е�����ֵ
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labelstmp
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

#ʹ�þ��������з���
#������ѵ���õľ����������Ա�ǩ��Ԥ�⼯������
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)   #index�������ҵ�ǰ�б��е�һ��ƥ��firstStr������Ԫ�ص�����
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

#�洢������
#������ѵ���õľ��������ļ���
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
