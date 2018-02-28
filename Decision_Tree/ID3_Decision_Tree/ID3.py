#!/usr/bin/env python3
# coding=utf-8
'''

                    ［ID3决策树算法］
   决策树算法作为机器学习经典算法之一，有着重要的实际意义
   从决策树中可以直观的得到数据的内在关系
   ID3决策树利用"信息增益"作为生成决策树的方法（属性划分的方法）
   本程序实现了ID3决策树的生成方法
   但还未实现决策树的剪枝

＊＊＊＊＊＊＊＊＊＊＊＊ID3决策树算法流程＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
    1.首先计算当前结点的信息熵
    2.按照不同属性划分数据集，得到信息增益
    3.选择信息增益最大的属性划分数据集
    4.对划分得到的每个数据集递归地重复1-3步，直到达到终止条件

    ＃决策树终止划分的条件有两个：当前结点全为一类 ； 无特征可再分，此时少数服从多数（投票）
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊

    为方便实现ID3决策树的生成算法，将程序拆解为四个模块：
    1.计算当前节点（数据集）的信息熵
    2.按照某属性划分数据集
    3.选择最优划分属性
    4.生成ID3决策树



'''

_author_ = 'zixuwang'
_datetime_ = '2018-2-27'


from numpy import *
from math import *
import time
import operator as op

# 首先生成本次测试所用的数据集,根据两个特征判断是否属于［鱼类］
def creatDataSet():
    dataSet = [[1,1,'Yes'],
               [1,1,'Yes'],
               [1,0,'No'],
               [0,1,'No'],
               [0,1,'No']]
    labels = ['不浮出水面是否可以生存','是否有脚蹼']

    return dataSet,labels

dataSet,labels = creatDataSet()

"""
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
                                        Part 1:计算数据集信息熵
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
"""

'''
首先计算数据集的信息熵，拿到的数据集特点是：最后一个属性是类别
计算信息熵的方法是：- Sum( p * log(p,2) )
'''
def getEntropy(dataSet):
    total_num = len(dataSet)    # 总共多少个样本
    entropy = 0.0
    label_counter = {}

    for row in dataSet:
        label_counter[row[-1]] = label_counter.get(row[-1],0) + 1   # 统计数据集中各类别的数量

    for label in label_counter:
        p = float(label_counter[label])/total_num
        entropy -=  p * log(p,2)    # 计算数据集的信息熵

    return entropy

# 此代码段测试：

# print(getEntropy(dataSet))

# dataSet[0][2] = 'maybe'
# print(getEntropy(dataSet))

# 正确结果应为：0.9709505944546686     ,   1.3709505944546687


"""
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
                                    Part 2:根据某属性划分数据集
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
"""
'''
根据某属性划分数据集，将数据集中满足条件的属性挑选出来（同时剔除该属性）
'''
def divideDataSet(dataSet,axis,value):
    returnDataSet = [] # 返回划分之后的数据集

    for row in dataSet:
        if row[axis] == value:  # 如果对应属性满足条件，则被挑选出来
            Vector = row[:axis]
            Vector.extend(row[axis+1:])
            returnDataSet.append(Vector)

    return returnDataSet

# 此代码段测试：

# print('划分之前：',dataSet)
# print('划分之后：',divideDataSet(dataSet,0,1))

# 正确结果应为：[[1, 'y'], [1, 'y'], [0, 'n']]


"""
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
                              Part 3:根据信息增益选取最优划分属性
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
"""

'''
首先遍历所有属性，用每个属性划分数据集并计算信息增益
计算信息增益的方法是： Sum（  (|D'| / |D|) * -Sum( p * log(p,2) )   ）
信息增益最大的那个就是所选择的划分属性
'''
def getBestAttribute(dataSet):
    bestFeature = -1
    maxInfoGain = 0.0
    base_entropy = getEntropy(dataSet)  #原始数据集信息熵

    feature_num = len(dataSet[0]) - 1   #最后一个是类别
    total_num = len(dataSet)     #得到样本总数

    for i in range(feature_num):    #遍历所有的属性
        total_entropy = 0.0
        infoGain = 0.0
        Vectors = [example[i] for example in dataSet]   #只保留原矩阵某属性对应的列
        uniqueVectors = set(Vectors)    #排除重复

        for feature in uniqueVectors:   #遍历该属性的每一个可能取值
            dividedDataSet = divideDataSet(dataSet,i,feature)   #按照该属性的取值划分数据集
            entropy = getEntropy(dividedDataSet)    #得到信息熵
            total_entropy  += len(dividedDataSet)/total_num * entropy    #得到总信息熵
            infoGain = base_entropy - total_entropy     #得到信息增益
        if infoGain > maxInfoGain:
            maxInfoGain = infoGain
            bestFeature = i

    return bestFeature


# 此代码段测试：

# print('最优划分属性：',getBestAttribute(dataSet))

# 正确结果应为：0


"""
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
                                    递归构造ID3决策树
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
"""
'''
由于构造决策树时，每次划分属性的选择方式完全相同，因此采用递归向下的方式生成决策树
但终止条件有两个：
    一个是当前集合中所有样本同属一类，这时候不用再划分
    其次是已经用完所有属性，再无属性可以使用，这个时候需要采用少数服从多数的投票法
    
先定义投票法的函数
'''
def vote(dataSet):
    labels = [example[-1] for example in dataSet]
    labelCounter = {}
    for label in labels:
        labelCounter[label] = labelCounter.get(label,0) + 1
    result = sorted(labelCounter.items(),key = op.itemgetter(1),reverse = True)
    return result[0][0]

# 此代码段测试：

# print('vote result：',vote(dataSet))

# 正确结果应为：   n

'''
ID3决策树生成
(labels-属性的名称是为了显示划分时的操作)
'''
def ID3(dataSet,labels):
    labelList = [example[-1] for example in dataSet]
    # 首先判断是否终止
    if labelList.count(labelList[0]) == len(labelList):     # 如果所有样本同属一类，终止
        return labelList[0]

    if len(dataSet[0]) == 1:        # 如果属性无剩余，终止
        return vote(dataSet)

    attribute = getBestAttribute(dataSet)   # 得到最优划分属性
    attributeLabel = labels[attribute]      # 得到最优划分属性的名称（便于存储、可视化等操作）

    DecisionTree = {attributeLabel:{}}      # 生成决策树
    del labels[attribute]                  # 删除这个已经使用过的属性

    featureList = [example[attribute] for example in dataSet]   # 得到该属性对应的全部取值
    featureSet = set(featureList)   # 排除重复部分

    for feature in featureSet:  # 对每个取值，产生独立的结点，并依次递归向下继续划分生成决策树
        subDataSet = divideDataSet(dataSet,attribute,feature)
        subLabels = labels[:]   #避免递归过程中改变原始列表（python中列表传参为引用）
        DecisionTree[attributeLabel][feature] = ID3(subDataSet,subLabels)

    return DecisionTree


# 此代码段测试：

# print('ID3 Decision Tree ：',ID3(dataSet,labels))

# 正确结果应为：  {'不浮出水面是否可以生存': {0: 'No', 1: {'是否有脚蹼': {0: 'No', 1: 'Yes'}}}}

