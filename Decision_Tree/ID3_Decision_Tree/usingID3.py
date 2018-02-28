#!/usr/bin/env python3
# coding=utf-8
'''

    ［使用ID3决策树］

包含以下内容：
1.如何保存生成的决策树模型
2.如何读取决策树模型
3.如何使用决策树对新来的样本进行分类
'''

_author_ = 'zixuwang'
_datetime_ = '2018-2-28'


"""
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
                                           Part 1:保存决策树模型
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
对于已经生成的决策树模型（一个dict），可以将其序列化并保存至硬盘上
"""
def storeTree(tree,filename='./DecisionTreeModel.mat'):
    import pickle
    file = open(filename, 'wb')
    pickle.dump(tree, file) #序列化并保存
    file.close()

"""
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
                                           Part 2:读取决策树模型
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
"""
def loadTree(filename):
    import pickle
    file = open(filename,'rb')
    return pickle.load(file)


"""
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
                                        Part 3:使用决策树模型分类
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
输入：决策树模型，所有特征名称，测试样本
"""

def classify(tree, featureLabels, sample):
    firstNode = list(tree.keys())[0]
    secondNode = tree[firstNode]
    featureIndex = featureLabels.index(firstNode)   #获得子树最顶端决策节点特征对应的索引（第几个特征）

    for kind in secondNode.keys():  #遍历该特征对应的所有取值
        if sample[featureIndex] == kind:    #找到与测试样本中该特征取值相符的特征
            if type(secondNode[kind]).__name__ == 'dict':
                label = classify(secondNode[kind],featureLabels,sample)
            else:
                label = secondNode[kind]

    return label


# 此代码段测试：

# from MachineLearning.Decision_Tree.ID3_Decision_Tree import ID3
# dataSet,labels = ID3.creatDataSet()
# featureLabels = labels[:]
# tree = ID3.ID3(dataSet,labels)  #这里有个bug找了好久，这个labels必须得复制一份，至于为啥前面说了
# print(tree)
# storeTree(tree,filename='./ID3Model.mat')
#
# model = loadTree('./ID3Model.mat')
# print(model)
# print(classify(model,featureLabels,[1,0]))
# print(classify(model,featureLabels,[1,1]))

# 正确结果应为：No   Yes




