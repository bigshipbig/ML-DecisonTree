# -*- coding: cp936 -*-
from math import log
import operator
#计算信息熵
#参数：数据集
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

#创建数据集
def createDataSet():
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return  dataSet , labels

#按照给定的特征划分数据
#参数：带待划分的数据集，划分数据集的特征，需要返回的特征的值
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:]) #a.extend(b)得到一个包含a,b所有元素的列表
            retDataSet.append(reduceFeatVec)#a.append(b)列表得到第四个元素为b
    return  retDataSet

#选择最好的数据集划分方式(以信息增益为标准)
#参数：数据集
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1   #特征个数
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):   #遍历特征第i个
        featureSet = set([example[i] for example in dataSet])   #第i个特征取值集合
        newEntropy= 0.0
        for value in featureSet:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)   #该特征划分所对应的entropy
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#投票表决属于哪一类
#参数：分类的列表
def majorityCnt(classList):
    classCount ={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#创建决策树
#参数：训练集，属性标签
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):    #类别完全相同则停止继续划分  返回类标签-叶子节点
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)       #遍历完所有的特征时返回出现次数最多的
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    #del(labels[bestFeat])
    labelstmp = [] #去除最优的属性
    for feat in labels:
        if feat != labels[bestFeat]:
            labelstmp.append(feat)
    featValues = [example[bestFeat] for example in dataSet]    #得到的列表包含所有的属性值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labelstmp
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

#使用决策树进行分类
#参数：训练好的决策树，属性标签，预测集的特征
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)   #index方法查找当前列表中第一个匹配firstStr变量的元素的索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

#存储决策树
#参数：训练好的决策树，文件名
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
