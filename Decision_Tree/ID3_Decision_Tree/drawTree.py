#!/usr/bin/env python3
# coding=utf-8
'''

     ［可视化决策树算法］
   利用Matplotlib可视化决策树
由于这部分代码比较便作图方向，就直接把书上代码搬来了
'''

_author_ = 'zixuwang'
_datetime_ = '2018-2-28'

'''
导入之前写好的ID3决策树生成算法
'''
from MachineLearning.Decision_Tree.ID3_Decision_Tree import ID3

'''
解决无法输出中文的问题
'''
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


'''
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝                     
                                    Part 1:利用注解工具画箭头注释
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

'''
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle='sawtooth',fc='0.8')   #决策节点／判断节点
leafNode = dict(boxstyle='round4',fc='0.8')         #叶子节点
arrow_args = dict(arrowstyle='<-')                  #箭头形状

#boxstyle = "swatooth"意思是注解框的边缘是波浪线型的，fc控制的注解框内的颜色深度

# 定义构造箭头的函数:  nodeText是箭头指向结点中的内容；sonPtr是注释框位置(子结点)；
#                      parentPtr是箭头起点位置(父结点)； nodeType是箭头指向结点的类型（叶子结点or判断结点）
def plotNodeTest(nodeText, sonPtr, parentPtr, nodeType):
    creatPlotTest.ax1.annotate(nodeText, xy=parentPtr, xycoords='axes fraction',
                                     xytext=sonPtr, textcoords='axes fraction',
                                     va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)


# 简单画几个箭头指向结点，测试一下代码是否正确
def creatPlotTest():
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axporps = dict(xticks=[], yticks=[])     #axporps作用时去掉横纵坐标
    creatPlotTest.ax1 = plt.subplot(111,frameon=False,**axporps)  #frameon=False 是指坐标轴没有边框线
    plotNodeTest('决策结点',(0.5, 0.1),(0.1,0.5),decisionNode)
    plotNodeTest('叶结点',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()

# creatPlotTest()

'''
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
                         Part 2：统计决策树中叶子节点的数量  和  树的深度／高度
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝



[这一部分代码其实挺重要的，属于数据结构的一些基本知识，利用了深度优先搜索]
'''
def getLeafNum(tree):
    leafNum = 0
    firstNode = list(tree.keys())[0]  #得到当前子树最顶上的判断节点
    secondNode = tree[firstNode]    #得到决策节点对应下一层的所有结点

    for kind in secondNode.keys():
        if type(secondNode[kind]).__name__  == 'dict':  #如果该节点是个子树，加上子树的叶结点
            leafNum += getLeafNum(secondNode[kind])
        else:                                           #如果该节点叶结点，叶结点数＝1
            leafNum += 1

    return leafNum

def getTreeDepth(tree):
    max_depth = 0
    firstNode = list(tree.keys())[0] #得到当前子树最顶上的判断节点
    secondNode = tree[firstNode]    #得到决策节点对应下一层的所有结点

    for kind in secondNode.keys():
        if type(secondNode[kind]).__name__ == 'dict':#如果该节点是个子树，该节点的当前深度为1+子树深度
            depth = 1 + getTreeDepth(secondNode[kind])

        else:                                       #如果该节点是个叶结点，深度为1
            depth = 1

        if depth > max_depth:                       #选取当前结点下深度最深的那个子树作为结点深度
            max_depth = depth

    return max_depth

# 下面生成数据集测试一下上面代码是否正确

def testData(i):
    tree = [{'不浮出水面是否可以生存': {0: 'No', 1: {'是否有脚蹼': {0: 'No', 1: 'Yes'}}}},
            {'不浮出水面是否可以生存': {0: 'No', 1: {'是否有脚蹼': {0:{'头部':{0:'No',1:'Yes'}}, 1: 'No'}}}}]

    return tree[i]

# 此代码段测试：

# print('叶结点数目：',getLeafNum(testData(0)),'树的深度：',getTreeDepth(testData(0)))
# print('叶结点数目：',getLeafNum(testData(1)),'树的深度：',getTreeDepth(testData(1)))
# dataSet,labels = ID3.creatDataSet()
# tree = ID3.ID3(dataSet,labels)
# print('叶结点数目：',getLeafNum(tree),'树的深度：',getTreeDepth(tree))



# 正确结果应为：  叶结点数目： 3 树的深度： 2
#                 叶结点数目： 4 树的深度： 3
#                 叶结点数目： 3 树的深度： 2


'''"""
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
                                           Part 3：可视化决策树
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

主要包括以下函数：
1.在箭头中间添加类别信息用以表示特征的取值
2.递归绘制子树
3.可视化决策树
'''

# 1.在箭头中间添加类别信息用以表示特征的取值
# sonPt是子结点，parentPt是父结点，text是特征的取值
def plotMidText(sonPt,parentPt,text,textSize):
    x = (sonPt[0] + parentPt[0]) / 2.0
    y = (sonPt[1] + parentPt[1]) / 2.0

    creatPlot.ax1.text(x, y, text, fontsize=textSize, color='red', rotation=-15)

# 2.递归绘制子树
# tree是需要绘制的子树，parentPtr是这棵子树父结点位置，nodeText是类别信息
# 其中，plotTree.totalWidth是全局变量，记录整棵树的宽度；plotTree.totalHight是全局变量，记录整棵树高度
#       plotTree.Xoff与plotTree.Yoff 是当前下一个需要绘制的结点的横纵坐标，持续改变
#       上面参数是为了让整棵树看起来居中
def plotTree(tree,parentPtr,nodeText,textsize):
    leafNum = getLeafNum(tree)
    treeDepth = getTreeDepth(tree)
    firstNode = list(tree.keys())[0]

    sonPtr = (plotTree.Xoff + (1.0+float(leafNum))/2.0/plotTree.totalWidth  ,  plotTree.Yoff)
    # 上面那行代码是为了找到子树顶点结点最合适的绘制位置

    plotMidText(sonPtr,parentPtr,nodeText,textsize)  # 标注箭头中间的类别信息
    plotNode(firstNode,sonPtr,parentPtr,decisionNode)   #绘制结点

    plotTree.Yoff =plotTree.Yoff - 1.0/plotTree.totalDepth    # 绘制的方法是深度优先搜索，所以接下来需要绘制下一层结点，因此要把纵坐标向下移动

    secondNode = tree[firstNode]
    for kind in secondNode.keys():
        if type(secondNode[kind]).__name__ == 'dict':   # 如果该结点是一棵子树，递归绘制
            plotTree(secondNode[kind],sonPtr,str(kind),textsize)
        else:                                           #如果该结点是叶结点，绘制叶结点
            plotTree.Xoff = plotTree.Xoff + 1.0/plotTree.totalWidth #即将绘制结点的X坐标(叶结点)
            plotNode(secondNode[kind],(plotTree.Xoff,plotTree.Yoff),sonPtr,leafNode)#绘制叶结点
            plotMidText((plotTree.Xoff,plotTree.Yoff),sonPtr,str(kind),textsize)#绘制类别信息

    # 由于绘制的方法是深度优先搜索，绘制完该结点的所有子结点后需要回到当前层，以便接下来绘制该结点的兄弟结点
    plotTree.Yoff =plotTree.Yoff + 1.0/plotTree.totalDepth


# 3.可视化决策树
# tree是需要绘制的子树，parentPtr是这棵子树父结点位置，nodeText是类别信息
# 其中，plotTree.totalWidth是全局变量，记录整棵树的宽度；plotTree.totalHight是全局变量，记录整棵树高度
#       plotTree.Xoff与plotTree.Yoff 是当前下一个需要绘制的结点的横纵坐标，持续改变
#       上面参数是为了让整棵树看起来居中

def creatPlot(tree,textsize):
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axporps = dict(xticks=[], yticks=[])  # axporps作用时去掉横纵坐标
    creatPlot.ax1 = plt.subplot(111, frameon=False, **axporps)  # frameon=False 是指坐标轴没有边框线

    #定义上述四个全局变量
    plotTree.totalWidth = float(getLeafNum(tree))
    plotTree.totalDepth = float(getTreeDepth(tree))
    plotTree.Xoff = - 0.5/plotTree.totalWidth
    plotTree.Yoff = 1.0

    plotTree(tree,(0.5,1.0),'',textsize)

    plt.show()


# 定义构造箭头的函数:  nodeText是箭头指向结点中的内容；sonPtr是注释框位置(子结点)；
#                      parentPtr是箭头起点位置(父结点)； nodeType是箭头指向结点的类型（叶子结点or判断结点）
def plotNode(nodeText, sonPtr, parentPtr, nodeType):
    creatPlot.ax1.annotate(nodeText, xy=parentPtr, xycoords='axes fraction',
                                     xytext=sonPtr, textcoords='axes fraction',
                                     va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)



#测试代码
# dataSet,labels = ID3.creatDataSet()
# creatPlot(ID3.ID3(dataSet,labels),15)
# creatPlot(testData(1),15)
# tree = testData(0)
# tree['不浮出水面是否可以生存'][2] = 'Maybe'
# creatPlot(tree,15)
