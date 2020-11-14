# Reference from the url: https://blog.csdn.net/asialee_bird/article/details/81118245

from math import log
import operator


# select_data[0] is the data
# select_data[1] is the index of the data
# select_data[2] is the index of the feature
# len(select_data[0][0]) is the number of the feature(add the label(yes/no))
class decision_tree:
    def __init__(self, \
                 select_data):
        super(decision_tree, self).__init__()
        self.feature_list = ['Age', 'Job', 'Marital', 'Education', 'Default', 'Housing', 'Loan', 'Contact', 'Month', 'Dayofweek', 'Campaign', 'Pdays', 'Previous', 'Poutcome', 'Emprate', \
                        'Consprice', 'Conscpnf', 'Euribor3m', 'Nremployed', 'label']
        self.select_data = select_data
        
        # self.labels = []
        # self.labels = self.label(self.select_data[2], self.feature_list)
        # self.shannonEnt = 0.0 
        # self.shannonEnt = self.calcShannonEnt(self.select_data[0])

        # self.chooseBestFeatureToSplit(self.select_data[0])
        


# 函数说明:返回数据集的分类属性
# Parameters:
#     label_idx - 数据集的分类属性索引
#     label_list - 数据集的分类属性列表
# Returns:
#     labels - 数据集的分类属性 
    def label(self, label_idx, label_list):
        labels = []
        for i in label_idx:
            labels.append(label_list[i])      
        return labels                                   #返回数据集的分类属性 


    def BubbleSort(self, change_idx, testVec):
        n = len(change_idx)
        if n <= 1:
            return change_idx, testVec
        for i in range (0, n):
            for j in range(0, n-i-1):
                if change_idx[j] > change_idx[j+1]:
                    (change_idx[j], change_idx[j+1]) = (change_idx[j+1], change_idx[j])
                    temp = testVec[:][j]
                    testVec[:][j] = testVec[:][j+1]
                    testVec[:][j+1] = temp
        return change_idx, testVec


# 函数说明:计算给定数据集的经验熵(香农熵)
# Returns:
#     shannonEnt - 经验熵(香农熵)
    def calcShannonEnt(self, dataSet):
        numEntires = len(dataSet)                      #返回数据集的行数
        labelCounts = {}                                 #保存每个标签(Label)出现次数的字典
        for featVec in dataSet:                        #对每组特征向量进行统计
            currentLabel = featVec[-1]                   #提取标签(Label)信息
            if currentLabel not in labelCounts.keys():   #如果标签(Label)没有放入统计次数的字典,添加进去
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1               #Label计数
        shannonEnt = 0.0                                 #经验熵(香农熵)
        for key in labelCounts:                          #计算香农熵
            prob = float(labelCounts[key]) / numEntires  #选择该标签(Label)的概率
            shannonEnt -= prob * log(prob, 2)            #利用公式计算
        return shannonEnt                                #返回经验熵(香农熵)


# 函数说明:按照给定特征划分数据集
# Parameters:
#     dataSet - 待划分的数据集
#     axis - 划分数据集的特征
#     value - 需要返回的特征的值
    def splitDataSet(self, dataSet, axis, value):
        retDataSet = []                                     #创建返回的数据集列表
        for featVec in dataSet:                             #遍历数据集
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]             #去掉axis特征
                reducedFeatVec.extend(featVec[axis+1:])     #将符合条件的添加到返回的数据集
                retDataSet.append(reducedFeatVec)
        return retDataSet                                   #返回划分后的数据集


# 函数说明:选择最优特征
# Parameters:
#     dataSet - 数据集
# Returns:
#     bestFeature - 信息增益最大的(最优)特征的索引值
    def chooseBestFeatureToSplit(self, dataSet):
        numFeatures = len(dataSet[0]) - 1                     #特征数量
        baseEntropy = self.calcShannonEnt(dataSet)                 #计算数据集的香农熵
        bestInfoGain = 0.0                                    #信息增益
        bestFeature = -1                                      #最优特征的索引值
        for i in range(numFeatures):                          #遍历所有特征
            #获取dataSet的第i个所有特征
            featList = [example[i] for example in dataSet]
            uniqueVals = set(featList)                         #创建set集合{},元素不可重复
            newEntropy = 0.0                                   #经验条件熵
            for value in uniqueVals:                           #计算信息增益
                subDataSet = self.splitDataSet(dataSet, i, value)           #subDataSet划分后的子集
                prob = len(subDataSet) / float(len(dataSet))           #计算子集的概率
                newEntropy += prob * self.calcShannonEnt(subDataSet)        #根据公式计算经验条件熵
            infoGain = baseEntropy - newEntropy                        #信息增益
            # print("第%d个特征的增益为%.3f" % (i, infoGain))             #打印每个特征的信息增益
            if (infoGain > bestInfoGain):                              #计算信息增益
                bestInfoGain = infoGain                                #更新信息增益，找到最大的信息增益
                bestFeature = i                                        #记录信息增益最大的特征的索引值
        return bestFeature                                             #返回信息增益最大的特征的索引值


# 函数说明:统计classList中出现此处最多的元素(类标签)
# Parameters:
#     classList - 类标签列表
# Returns:
#     sortedClassCount[0][0] - 出现此处最多的元素(类标签)
    def majorityCnt(self, classList):
        classCount = {}
        for vote in classList:                                        #统计classList中每个元素出现的次数
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)        #根据字典的值降序排序
        return sortedClassCount[0][0]                                #返回classList中出现次数最多的元素


# 函数说明:递归构建决策树
# Parameters:
#     dataSet - 训练数据集
#     labels - 分类属性标签
#     featLabels - 存储选择的最优特征标签
# Returns:
#     myTree - 决策树
    def createTree(self, dataSet, labels, featLabels):
        classList = [example[-1] for example in dataSet]               #取分类标签(是否放贷:yes or no)
        if classList.count(classList[0]) == len(classList):            #如果类别完全相同则停止继续划分
            return classList[0]
        if len(dataSet[0]) == 1:                                       #遍历完所有特征时返回出现次数最多的类标签
            return self.majorityCnt(classList)
        bestFeat = self.chooseBestFeatureToSplit(dataSet)                   #选择最优特征
        bestFeatLabel = labels[bestFeat]                               #最优特征的标签
        featLabels.append(bestFeatLabel)
        myTree = {bestFeatLabel:{}}                                    #根据最优特征的标签生成树
        del(labels[bestFeat])                                          #删除已经使用特征标签
        featValues = [example[bestFeat] for example in dataSet]        #得到训练集中所有最优特征的属性值
        uniqueVals = set(featValues)                                   #去掉重复的属性值
        for value in uniqueVals:
            subLabels=labels[:]
            #递归调用函数createTree(),遍历特征，创建决策树。
            myTree[bestFeatLabel][value] = self.createTree(self.splitDataSet(dataSet, bestFeat, value), subLabels, featLabels)
        return myTree


# 函数说明:使用决策树执行分类
# Parameters:
#     inputTree - 已经生成的决策树
#     featLabels - 存储选择的最优特征标签
#     testVec - 测试数据列表，顺序对应最优特征标签
# Returns:
#     classLabel - 分类结果
    def classify(self, inputTree, featLabels, testVec):
        classLabel = 'no'
        firstStr = next(iter(inputTree))             #获取决策树结点
        secondDict = inputTree[firstStr]             #下一个字典
        featIndex = featLabels.index(firstStr)
        for key in secondDict.keys():
            # print(key)
            # print(testVec[featIndex])
            if testVec[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = self.classify(secondDict[key], featLabels, testVec)
                else:
                    classLabel = secondDict[key]
        return classLabel
