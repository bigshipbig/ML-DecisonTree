from decisiontree import *
print "|***** create dataset *****|"
dataSet ,labels = createDataSet()
print "|***** original dataset*****|"
print dataSet
print labels
print "|***** the original shannorent *****|"
print calcShannonEnt(dataSet)
print  "|******test splitdataset *****|"
print splitDataSet(dataSet,0,0)

print "|***** testing chooseBestFeature *****|"
print chooseBestFeatureToSplit(dataSet)

print "|***** creating the decisionTree *****|"
myTree = createTree(dataSet,labels)
print myTree
print classify(myTree,labels,[1,0])
print classify(myTree,labels,[1,0])
print classify(myTree,labels,[1,1])

storeTree(myTree,"./txt/fishclassfy.txt")
print  grabTree("./txt/fishclassfy.txt")


