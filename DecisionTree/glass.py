from decisiontree import *
fr = open("./txt/lenses.txt")
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lenseslabels = ['age','prescript','astigmatic','tearRate']
lensesTree = createTree(lenses,lenseslabels)
print lensesTree