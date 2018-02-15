[KNN：K-近邻算法]
-----
<br>
K-近邻算法应该算是机器学习中最简单的算法了<br>
同时，K近邻算法属于懒惰学习算法，所谓懒惰学习，就是学习器不需要预先训练，而只在碰到测试样本时才会使用训练集<br>
也就是说，懒惰学习在训练阶段仅仅是把样本保存起来,训练时间开销为零,待收到测试样本后再进行处理<br>
<br>
<br>
<br>
KNN算法的步骤如下：<br>
<br>
1.数据收集和数据预处理、归一化等<br><br>
2.输入测试样本，计算与其他样本的距离<br><br>
3.对距离进行排序，选择最近的 K 个样本<br><br>
4.对这K个样本进行投票统计，选取票数最高的label作为测试样本的预测<br><br>
<br>
<br>
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝<br>
下述文件中<br>
KNN_Classifier.py从最简单的KNN分类器引入<br>
KNN_dating.py利用KNN算法判断约会对象是否符合心意<br>
最后KNN_handWritting.py利用KNN算法改进了手写体识别过程<br>
<br>
<br>
<br>
tips:<br>
foxmail：  zixuwang1997@foxmail.com<br>
gamil:     zixuwang1997@gmail.com<br>
others:    zixuwang@csu.edu.cn<br>
