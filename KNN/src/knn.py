import operator

import numpy as np


# 创建训练数据
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", 'A', "B", "B"]
    return group, labels


# 进行分类
def classfy(inX, dataset, labels, k):
    # 求距离
    dataSetSize = dataset.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataset
    sqDiffMat = diffMat ** 2
    sqDistace = sqDiffMat.sum(axis=1)
    distace = sqDistace ** 0.5
    # 将距离从小到大排序，argsort方法，将数组中的元素从小到大排列，返回依次返回其原来的索引
    sortDistIndicies = distace.argsort()
    # 使用字典来存储每个标签出现的次数，因为字典的键是不能重复的。
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        # 刚好可以获取当前距离对应的标签，操作真的秀啊
        voteLabel = labels[sortDistIndicies[i]]
        # 字典的键不能重复，刚好可以吧重复的次数记录下来
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    # 按照标签出现的次数从大到小排序，按照字典的值的大小对字典的键值对进行排序，ture为从大到小排，变成list了。
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回的是出现次数最多的一个标签，
    return sortedClassCount[0][0]


# 解析数据
def parseFile(filePath):
    # 打开文件
    fr = open(filePath)
    # 多行读取文件
    arrayLines = fr.readlines()
    # 获取行数
    numberOfLines = len(arrayLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayLines:
        lines = line.strip()
        listFormLine = lines.split(",")
        returnMat[index, :] = listFormLine[0:3]
        classLabelVector.append(int(listFormLine[-1]))
        index = index + 1
    return returnMat, classLabelVector

#数据归一化
def autoNorm(dataSet):
    minValues = dataSet.min(0)
    maxValues = dataSet.max(0)
    range = maxValues - minValues
    normData = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    # title函数的使用，通过重复第一个参数来构造数组，重复的次数为第二个参数，
    # 第二个参数为一维的则构造的数组为一维的，如果重复的参数为二维的则构造的数组为二维的
    # 行向重复m次，列向重复1次，
    normData = dataSet - np.tile(minValues, (m, 1))
    normData = normData / np.tile(range, (m, 1))
    return normData,range,minValues


