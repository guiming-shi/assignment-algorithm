# -*- coding: utf-8 -*-
__author__ = 'guiming'

import data_process as process
import decision_tree as dt
import pandas as pd
from plot_tree import createPlot
import process as ps

train_number = 10000
train_feature_number = 2

def cal_precision(testVec, label):
    precision = 0
    for i in range(len(testVec)):
        test_data = testVec[i]
        result = model.classify(myTree, featLabels, test_data)
        if result == label[i]:
            precision = precision + 1
    precision = precision / len(testVec)
    return precision

if __name__ == '__main__':
#----------------------得到训练样本和测试样本
    data = pd.read_csv("bank-additional-full.csv")
    bank_list = data.values.tolist()
    data_processor = process.data_process(bank_list=bank_list)
    data = data_processor.select(train_number, train_feature_number)
#----------------------生成决策树
    model = dt.decision_tree(select_data = data)
    featLabels = []
    myTree = model.createTree(dataSet = model.select_data[0], labels = model.label(model.select_data[2], model.feature_list), featLabels = featLabels)
    featLabels = featLabels[:train_feature_number]
    # print(myTree)
    createPlot(myTree)                    #画出决策树
#----------------------测试决策树效果
    testVec = data[3][:-1]
    label = [data[3][i][-1] for i in range(len(data[3]))]
    
    change_idx = []
    change_idx = ps.change_idx(featLabels)

    change_idx, testVec = ps.BubbleSort(change_idx, testVec)

    precision = cal_precision(testVec, label)
    print('precision: ', precision)




    



