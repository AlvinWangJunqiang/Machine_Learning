#!/usr/bin/env python3
# coding=utf-8
'''

             ［KNN算法］
    K-近邻算法是典型的懒惰学习算法
    所谓懒惰学习，就是学习器不需要预先训练，而只在碰到测试样本时才会使用训练集
    懒惰学习在训练阶段仅仅是把样本保存起来,训练时间开销为零,待收到测试样本后再进行处理

＊＊＊＊＊＊＊＊＊＊＊＊KNN算法流程＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
    1.数据收集和数据预处理、归一化等
    2.输入测试样本，计算与其他样本的距离
    3.对距离进行排序，选择最近的 K 个样本
    4.对这K个样本进行投票统计，选取票数最高的label作为测试样本的预测
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊


'''

_author_ = 'zixuwang'
_datetime_ = '2018-2-14'

from numpy import *
import operator as op
import matplotlib
import matplotlib.pyplot as plt


'''
为方便测试KNN算法，我们首先主动生成一个数据集和label
'''
def creatDataSet():
    DataSet = array([[0.1,0.1],
                     [0.2, 0.1],
                     [0.1, 0.2],
                     [0.1, 0.3],
                     [1.1, 1.1],
                     [1.1, 1.0],
                     [1.2, 1.5]])
    labels = ['-','-','-','-','+','+','+']

    return DataSet,labels

'''
进行数据可视化
数据可视化可以检查异常数据
同时可以直观的理解数据本身
当然，当数据特征维度过高时无法直接可视化
可以选择使用PCA等降维方式降维后可视化
'''
def showData(DataSet,labels):
    fig = plt.figure()  # 创建一张图
    ax = fig.add_subplot(111)   #  共1行、1列、属于（从左到右从上到下）第1块
    ax.scatter(DataSet[0:4,0],DataSet[0:4,1])
    ax.scatter(DataSet[4:7,0],DataSet[4:7,1],marker='+')
    #Matplotlib scatter：https://www.cnblogs.com/shanlizi/p/6850318.html
    plt.show()

# 检查数据
Data,label = creatDataSet()
print(Data)
print(label)
showData(Data,label)

'''
KNN分类器
输入：（输入数据，训练数据集，训练集类别，k）
输出： 预测类别
'''
def KNN_Classifier(inputData,DataSet,labels,k):
    data_num = DataSet.shape[0]     #得到训练数据的样本总量
    inputDatas = tile(inputData,(data_num,1))       #tile函数是变换函数，可以把inputData复制n次
                                                    #（n，1）的意思是复制n行1列
                                                    #更多tile函数变换方法可见https://www.cnblogs.com/100thMountain/p/4719113.html

    # 下面这四步是算输入数据和训练数据所有特征的欧式距离，也就是每一行向量的L2范式
    matrix_reduced = DataSet - inputDatas
    matrix_pow2 = matrix_reduced ** 2
    matrix_powSum = matrix_pow2.sum(axis=1)     #  axis＝1是矩阵每一行求和；axis＝0是矩阵每一列求和
    matrix_distance = matrix_powSum ** 0.5      #  得到欧式距离

    # 下面几步是求得排名前3的label
    label_dict = {}    # 创建Map，统计排名前三的labels
    index_distance = matrix_distance.argsort()    #  排序,由于labels与matrix一一对应，因此不能直接sorted

    for i in range(k):
        K_label = labels[index_distance[i]]   # 得到排名第k的类别
        label_dict[K_label] = label_dict.get(K_label,0) + 1 #在字典中得到统计，如果没有这一项的话默认为0

    # sorted_labels = sorted(label_dict.items(),key = op.itemgetter(1),reverse=True)  #对得到的前k个label进行降序排序
   # 上面有个很神奇的bug：https://stackoverflow.com/questions/24463202/typeerror-get-takes-no-keyword-arguments

    sorted_labels = sorted(label_dict.items(),key = op.itemgetter(1),reverse=True)  #对得到的前k个label进行降序排序
    return sorted_labels[0][0]  #返回出现次数排名第一的label


'''
检验kNN算法
'''
predict = KNN_Classifier([0.9,0.8],Data,label,3)
print('[0.9,0.8]------->',predict)
