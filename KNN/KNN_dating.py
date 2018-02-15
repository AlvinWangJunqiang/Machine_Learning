#!/usr/bin/env python3
# coding=utf-8
'''

     ［KNN算法应用：约会网站匹配改进］
    利用K近邻算法完成一个具体的项目：约会网站匹配改进
    Alice在社交网站上经常会匹配感兴趣的男性朋友，但有时候她对这些匹配到的人并不感兴趣
    她收集了很多匹配到的男性朋友的三个特征：［飞行里程数、吃冰激凌的公升数、打电子游戏的时间比例］
    按照这三个特征，可以将它们分为三类：（不喜欢，一般喜欢，很喜欢）

    本实例中利用KNN算法，对新的样本来做分类
    在本例中，主要是演示一下如何进行数据集的预处理，如何使用KNN分类器

＊＊＊＊＊＊＊＊＊＊＊＊KNN算法流程＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
    1.数据收集和数据预处理、归一化等
    2.输入测试样本，计算与其他样本的距离
    3.对距离进行排序，选择最近的 K 个样本
    4.对这K个样本进行投票统计，选取票数最高的label作为测试样本的预测
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊


'''

_author_ = 'zixuwang'
_datetime_ = '2018-2-15'

file_path = './DataSet/datingTestSet2.txt'
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator as op

'''
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
                                     读 取 文 件 数 据 集
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
'''
'''
拿到一个数据集，首先要对它做处理，将其转换为KNN分类器可以操作的数据类型（矩阵）
'''


def File2Matrix(file_path):
    file = open(file_path)
    fileLines = file.readlines()
    data_num = len(fileLines)
    data_matrix = zeros((data_num, 3))
    data_labels = []

    index = 0
    for line in fileLines:
        line = line.strip()  # 去除字符串头尾的空格
        line2array = line.split('\t')
        data_matrix[index, :] = line2array[0:-1]
        data_labels.append(int(line2array[-1]))  # 得到数据对应的类别，注意！需要转换为int，否则默认为strinf
        index = index + 1

    return data_matrix, data_labels

'''
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
'''

'''
检查数据导入是否正确
'''
original_Data, labels = File2Matrix(file_path)
print()
print('original_Data')
print(original_Data)
print('labels')
print(labels[:20])

'''
数据可视化，再次检查数据
'''


def showData(DataSet, labels):
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.scatter(DataSet[:, 0], DataSet[:, 1], 15.0 * array(labels), 15.0 * array(labels))
    ax2 = fig.add_subplot(312)
    ax2.scatter(DataSet[:, 0], DataSet[:, 2], 15.0 * array(labels), 15.0 * array(labels))
    ax3 = fig.add_subplot(313)
    ax3.scatter(DataSet[:, 1], DataSet[:, 2], 15.0 * array(labels), 15.0 * array(labels))
    plt.show()


showData(original_Data, labels)



'''
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
                                        数 据 归 一 化 处 理
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
'''
'''
数据归一化处理
由于不同维度的特征有不同的取值范围，我们不能说数值打的影响就大
也就是说不能让不同维度的特征因为其单位不同而对分类产生不同的影响
此时就应该对原始数据进行归一化处理，常见的有0均值归一化、0均值单位方差归一化等
这里我们采用（0，1）范围归一化：
    normalized_Data ＝ （original_Data － Min） ／ （Max－Min）
'''


def DataNormalize(original_Data):
    data_num = original_Data.shape[0]
    Min = original_Data.max(0)  # 获得每一列的最大值（也就是每个维度特征的最大值）
    Max = original_Data.min(0)
    ranges = Max - Min

    normalized_Data = (original_Data - tile(Min, (data_num, 1))) / tile(ranges, (data_num, 1))
    return normalized_Data, Min, ranges  # 注意，需要返回range和Min！！！！

'''
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
'''



normalized_Data,_,_ = DataNormalize(original_Data)
print()
print('normalized_Data')
print(normalized_Data)
showData(normalized_Data, labels)



'''
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
                                        KNN 分 类 器
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
'''
'''
下面是KNN分类器,直接导入刚才写好的KNN分类器即可
'''
def KNN_Classifier(inputData,DataSet,labels,k):
    data_num = DataSet.shape[0]
    reduce_matrix = DataSet - tile(inputData,(data_num,1))
    pow2_matrix = reduce_matrix ** 2
    sum_matrix = pow2_matrix.sum(axis=1)
    distance_matrix = sum_matrix ** 0.5 #得到距离
    distance_index = distance_matrix.argsort()

    label_dict = {}
    for i in range(k):
        k_label = labels[distance_index[i]]
        label_dict[k_label] = label_dict.get(k_label,0) + 1

    label_sorted = sorted(label_dict.items(),key=op.itemgetter(1),reverse=True)

    return label_sorted[0][0]


'''
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
                                        精 度 测 试
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
'''
'''
接下来进行精度测试，看一看KNN分类器效果如何
一般来说我们要把数据集分为训练集、验证集、测试集三类
训练集用来训练模型、验证集用来调整模型和调节参数、测试集用来检验精度
由于KNN是懒惰学习器，我们只需要训练集和测试集即可
'''


def KNN_accurary(testSet_rate):
    # 首先导入数据集，进行预处理和归一化
    original_Data, labels = File2Matrix(file_path)
    normalized_Data, Min, ranges = DataNormalize(original_Data)
    testSet_num = int(testSet_rate * normalized_Data.shape[0])
    trainingSet = normalized_Data[testSet_num:, :]  #在整体数据集中除去前testSet_num个，其余作为训练数据集  print(trainingSet.shape)
    correct_num = 0

    for i in range(testSet_num):
        # 由于测试数据集本身已经归一化，因此不需要归一化的步骤了
        if (labels[i] == KNN_Classifier(normalized_Data[i,:], trainingSet, labels[testSet_num:], 3)):
            correct_num += 1

    print('测试集上的精度为：%.2f%%' % (correct_num / testSet_num * 100))



'''
检查精度是否符合要求
'''
KNN_accurary(0.10)


'''
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
                                        使用KNN分类器做分类
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
'''


'''
经过了精度验证，再碰到具体的样本时，应该如何进行处理呢？
'''
def using_KNN(inputData):
    # 先读取数据，进行归一化
    orignial_data,labels = File2Matrix(file_path)
    normalized_Data,Min,ranges = DataNormalize(original_Data)
    inputData = (inputData - Min) / ranges  #！！！注意：一定要对输入样本进行归一化处理
    predict = KNN_Classifier(inputData,normalized_Data,labels,3)
    return predict


def main():
    #  raw_input和input的区别： https://www.cnblogs.com/yunquan/p/6950685.html
    gamingTime = float(input('打游戏所占的时间比例（％）：'))
    flyMiles = float(input('每年飞行的公里数：'))
    iceCream = float(input('每年吃的冰激凌有多少升：'))
    inputData = [flyMiles,gamingTime,iceCream]
    predict = using_KNN(inputData)
    kinds = ['不感兴趣','可能有一点点兴趣','非常感兴趣']
    print('你对这个人感兴趣的程度可能是： %s'%(kinds[predict - 1]))


main()