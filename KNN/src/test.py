import os

import numpy as np

os.chdir(r"F:\baincheng\python\MachineLearning\KNN\src")
import knn

# 定义训练样本集，和标签，

# 读取数据
#dateData, type = knn.readFile("F:\\baincheng\\python\\MachineLearning\\KNN\data\\dateData.txt")
# 归一化数据
# normedData, ranges, minvalue = knn.autoNorm(dateData)
# 分类
#type1 = knn.classfy([568, 7, 1], normedData, type, 8)

# print(type1)

# 测试
# a = knn.testClassify("F:\\baincheng\\python\\MachineLearning\\KNN\data\\dateData.txt")
# print(a)

# 正式进行预测
def classifyPerson():
    resultList = ["not at all", "in small doses", "in large doses"]
    #使用控制台输入
    precentTats = float(input("precentage of time spend playing vido game"))
    ffMiles = float(input("frequent fier miles earned per year"))
    iceCream = float(input("liters of icecream consume per yaer"))
    dateData, typeLabels = knn.readFile("F:\\baincheng\\python\\MachineLearning\\KNN\\data\\dateData.txt")
    normedData, ranges, minvalues = knn.autoNorm(dateData)
    inArr = np.array([ffMiles, precentTats, iceCream])
    classifierResult = knn.classfy(inArr, normedData, typeLabels, 5)
    print("you will probably this person :", resultList[classifierResult - 1])


classifyPerson()
# 散点图显示数据
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dateData[:, 0], dateData[:, 1],15.0*np.array(type),1.5*np.array(type))
# plt.show()
