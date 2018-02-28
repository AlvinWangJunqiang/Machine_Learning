#!/usr/bin/env python3
# coding=utf-8
'''
    [利用ID3决策树进行隐形眼镜分类]
'''

_author_ = 'zixuwang'
_datetime_ = '2018-2-28'

# 读取数据集和特征标签
def loadDataSet():
    file = open('./lenses.txt')
    dataSet = [vector.strip().split('\t') for vector in file]
    labels = ['age','prescript','astigmatic','tearRate']
    #年龄、药方？、散光性、眼泪率
    return dataSet,labels

from MachineLearning.Decision_Tree.ID3_Decision_Tree import ID3
from MachineLearning.Decision_Tree.ID3_Decision_Tree import drawTree

dataSet,labels = loadDataSet()
tree = ID3.ID3(dataSet,labels)
drawTree.creatPlot(tree,textsize=10)
