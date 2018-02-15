#!/usr/bin/env python3
# coding=utf-8
'''

            ［KNN算法应用：手写数字识别］
    利用K近邻算法完成一个具体的项目：手写数字识别
    当我们碰到手写字体时，可以利用KNN分类起来进行识别／分类
    在这个例子中可以看到，KNN分类器的精度很高，错误率理论上不高于两倍最优贝叶斯误差
    但是由于KNN分类器是懒惰学习，每次都要适用所有的训练集，因此速度很慢，占用空间很大

    本例中的输入信息为图片，为方便操作，使用txt文件保存（这显然不是一个明智的做法，因为txt文件需要转码）
    观察一下DataSet中的digits的文件中的txt文件，可以看出，数据集被分为训练数据集和测试数据集
    同时我们可以从每一个txt文件的名字中得到类别信息


    本实例中利用KNN算法，完成手写数字识别

＊＊＊＊＊＊＊＊＊＊＊＊KNN算法流程＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
    1.数据收集和数据预处理、归一化等
    2.输入测试样本，计算与其他样本的距离
    3.对距离进行排序，选择最近的 K 个样本
    4.对这K个样本进行投票统计，选取票数最高的label作为测试样本的预测
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊



'''

_author_ = 'zixuwang'
_datetime_ = '2018-2-15'


path = './DataSet/digits'

from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
import time
import operator as op
import random

'''
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
                                     读 取 文 件 数 据 集
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
'''

'''
对于一般的二维图片，我们的操作一般是将一幅图片处理成一个行向量
这样可以用一个m*n的矩阵储存所有的手写体
由于输入数据的特征只有0和1两种类别，所以不用进行标准化
'''
# 这个函数用来把一副图片（虽然是txt文件,尺寸 32x32 ）转换成一个行向量
def Image2Vector(file):
    imageVector = zeros((1,1024))
    image = open(file)
    for i in range(32):
        imageLine = image.readline()
        for j in range(32):
            imageVector[0, 32*i+j] = int(imageLine[j])
    return imageVector

# 这个函数用来读取训练数据集，保存到一个矩阵中
def getDataMatrix(type):
    if type=='train' :  type = 'trainingDigits'
    else:type = 'testDigits'

    trainingImageList = listdir('%s/%s'%(path,type))  #得到文件夹中所有的文件目录
    trainingImage_num = len(trainingImageList)
    trainingMatrix = zeros((trainingImage_num,1024))
    trainingLabels = []

    for i in range(trainingImage_num):
        imageName = trainingImageList[i]
        # if i==0:print('%s/%s/%s'%(path,type,imageName))  #可以检查文件路径是否正确
        trainingMatrix[i,:] = Image2Vector('%s/%s/%s'%(path,type,imageName)) #将图片转换成行向量

        label = int(imageName.split('.')[0].split('_')[0])   #从文件名中得到该手写体对应的label
        trainingLabels.append(label)

    return trainingMatrix,trainingLabels


# 这个函数用来可视化数据
def showData(ImageVector):
    ImageMatrix = ImageVector.reshape([32,32])
    plt.figure(figsize=(3, 3))
    plt.axis("off")
    plt.imshow(ImageMatrix)
    plt.show()


'''
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
'''

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
                                     使用KNN分类器验证精度
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
'''
def KNN_accurary():
    time1 = time.time()
    TrainingMatrix,TrainingLabels = getDataMatrix('train')
    TestMatrix,TestLabels = getDataMatrix('test')
    reading_cost = time.time() - time1

    correct_times = 0.0
    total_times = len(TestLabels)
    time2 = time.time()
    for i in range(total_times):
        predict =  KNN_Classifier(TestMatrix[i,:],TrainingMatrix,TrainingLabels,3)
        if predict == TestLabels[i]:    correct_times += 1.0

    predict_cost = time.time() - time2
    print("测试集上的精度为：%.2f%%"%(correct_times/total_times * 100))
    print('读取数据耗时为： %.2fs'%reading_cost)
    print('精度验证耗时为： %.2fs'%predict_cost)

'''
精度验证可以看到：

测试集上的精度为：98.94%
读取数据耗时为： 10.62s
精度验证耗时为： 22.54s

两倍最优贝叶斯错误率不是盖的
就是速度不快，而且这个模型还得占用很大的空间
'''
# KNN_accurary()


'''
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
                                     使用KNN识别手写体
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
'''
def KNN_predict():
    TrainingMatrix, TrainingLabels = getDataMatrix('train')
    TestMatrix, TestLabels = getDataMatrix('test')

    total_num = len(TestLabels)

    for i in range(total_num):
        i = int(random.random()*1000)
        if i > total_num : i = i - total_num
        predictLabel = KNN_Classifier(TestMatrix[i,:],TrainingMatrix,TrainingLabels,3)
        trueLabel = TestLabels[i]
        print()
        print('KNN预测类别为：',predictLabel)
        print('实际类别为：',trueLabel)
        showData(TestMatrix[i, :])


KNN_predict()
