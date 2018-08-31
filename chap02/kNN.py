# -*- coding:utf-8 -*-
from numpy import *
import operator
from os import listdir
def createDataSet():
    group = array([[1, 1.1], [1, 1], [0, 0], [0, 0.1]])#创建数据集
    labels = ['A', 'A', 'B', 'B']#创建标签
    return group, labels

# KNN分类
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
    # sorted返回一个新的 而sort是在原来的基础上调整
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 将txt文件里的数据提取到内存中
def file2matrix(filename):
    fr = open(filename)# 打开文件
    arrayOLines = fr.readlines()# 文件中的每一行作为一个字符串，所有字符串组成一个list
    numberOfLines = len(arrayOLines)# 得到原文件的行数
    returnMat = zeros((numberOfLines, 3))# 初始化特征矩阵，特征数是3
    classLabelVector = []# 初始化类别列别
    index = 0
    # 解析文件数据到列表
    for line in arrayOLines:
        line = line.strip()# 删除字符串开头和结尾的空白符
        listFromLine = line.split('\t')# 按‘\t’字符来分割原来的字符串，返回一个list
        returnMat[index, :] = listFromLine[0:3]# 将前3列的特征取出
        classLabelVector.append(int(listFromLine[-1]))# 取最后一列（类别号）存入
        index += 1
    return returnMat, classLabelVector

# 数据归一化 newValue = (oldValue - min) / (max - min)
def autoNorm(dataSet):
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    ranges = maxVal - minVal
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVal, (m,1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVal

# 分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10# 测试集比例
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')# 读取数据
    normMat, ranges, minVal = autoNorm(datingDataMat)# 归一化特征值
    m = normMat.shape[0]# 样本总数
    numTestVecs = int(m*hoRatio)# 用作测试的样本数目
    errorCount = 0.0# 初始化错误
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("分类器预测值: %d, 真实值: %d"%(classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("预测错误率为: %f" % (errorCount/float(numTestVecs)))

# 约会网站预测函数
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    persentTats = float(input("percentage of time spent playing video games? "))
    ffMiles = float(input("frequent flier miles per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, persentTats, iceCream])
    classfierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print("you will probably like this person: ", resultList[classfierResult - 1])

# ------------------------------2.3 手写识别系统----------------------------------

# 将图像(保存在txt)格式化处理为一个向量
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):# 遍历每一行
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    traingFileList = listdir('trainingDigits')# 列出给定目录的文件名
    m = len(traingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = traingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classfierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("分类预测值：%d, 真实值：%d" % (classfierResult, classNumStr))
        if(classfierResult != classNumStr):
            errorCount += 1.0
    print("预测错误数：%f" % errorCount)
    print("预测错误率：", errorCount/float(mTest))




