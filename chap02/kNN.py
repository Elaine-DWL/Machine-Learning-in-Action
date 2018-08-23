from numpy import *
import operator
def createDataSet():
    group = array([[1, 1.1], [1, 1], [0, 0], [0, 0.1]])#创建数据集
    labels = ['A', 'A', 'B', 'B']#创建标签
    return group, labels

def classify0(inX, dataSet, labels, k):# inx 是待分类的某个样本，dataset是类别已知的训练数据
    dataSetSize = dataSet.shape[0]# 训练集样本个数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet# 计算inx与dataset中每个样本的差异（对应特征相减）
    sqDiffMat = diffMat**2# 特征差的平方
    sqDistances = sqDiffMat.sum(axis = 1)# 所有特征差的平方求和，作为最后距离的平方
    distances = sqDistances**0.5# 对距离进行开方   得到一个列向量
    # sortedDistIndices = distances.argsort()#返回数组值从小到大的索引
    sortedDistIndices = argsort(distances)
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # sorted返回一个新的
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)# 打开文件
    arrayOLines = fr.readlines()# 文件中的每一行作为一个字符串，所有字符串组成一个list
    numberOfLines = len(arrayOLines)# 得到原文件的行数
    returnMat = zeros(numberOfLines, 3)
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        return
