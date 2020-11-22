# -*- coding: utf-8 -*-
__author__ = 'guiming'

import data_process as process
import decision_tree as dt
import pandas as pd
from plot_tree import createPlot
import process as ps

train_number = 10000
train_feature_number = 2
tree_number = 100
threshold = tree_number//1.2

def cal_precision(testVec, label, model_idx):
    precision = 0
    for idx in range(len(testVec)):
    # for idx in range(1):   
        test_data = testVec[idx]
        result_tree = model_list[model_idx].classify(myTree[model_idx], featLabels[model_idx], test_data)
        if result_tree == label[model_idx][idx]:
            precision = precision + 1
        result[idx][model_idx] = result_tree
    precision = precision / len(testVec)
    return precision

if __name__ == '__main__':
#----------------------load数据和进行处理
    data = pd.read_csv("bank-additional-full-train.csv")
    bank_list = data.values.tolist()
    data_processor = process.data_process(bank_list=bank_list)
#----------------------生成决策树
    model_list = []
    featLabels = []
    myTree = []
    data = []
    for i in range(tree_number):
        data.append(data_processor.select(train_number, train_feature_number))
        model_list.append(dt.decision_tree(select_data = data[i]))
        featLabels.append([])
        model = model_list[i]
        myTree.append(model.createTree(dataSet = model.select_data[0], labels = model.label(model.select_data[2], model.feature_list), featLabels = featLabels[i]))
        featLabels[i] = featLabels[i][:train_feature_number]
        createPlot(myTree[i])                    #画出决策树
#----------------------测试集数据
    data_test = pd.read_csv("bank-additional-full-test.csv")
    bank_list_test = data_test.values.tolist()
    data_processor_test = process.data_process(bank_list=bank_list_test)

    testVec = [[]] * tree_number
    label = [[]] * tree_number
    for i in range(tree_number):
        testVec[i] = data_processor_test.get_test_data(data[i][2])
        label[i] = data_processor_test.get_test_label()
#----------------------测试测试集决策树效果
    for model_idx in range(tree_number):
        # print('-------------')
        change_idx = []
        change_idx = ps.change_idx(featLabels[model_idx])

        change_idx, testVec[model_idx] = ps.BubbleSort(change_idx, testVec[model_idx])

    result = []
    for model_idx in range(tree_number):   
        result_tmp = []
        precision = 0
        for idx in range(len(testVec[model_idx])):
            test_data = testVec[model_idx][idx]
            result_pre = model_list[model_idx].classify(myTree[model_idx], featLabels[model_idx], test_data)
            if result_pre == label[model_idx][idx]:
                precision = precision + 1
            result_tmp.append(result_pre)
        result.append(result_tmp)
        precision = precision / len(testVec[model_idx])   

        # print('precision tree', model_idx, ': ', precision)


    TP_list = []
    FP_list = []
    FN_list = []
    for thres in [100,20,10,8,7,5,3,2,1.8,1.5,1.3,1.1,1]:
        threshold = tree_number / thres
#----------------------oob选择随机森林组成方式N_list.append(FN)
        pre_list = []
        for j in range(len(result[0])):
            pre = 0
            for model_idx in range(len(result)):
                if result[model_idx][j] == 'yes':
                    pre = pre + 1
            if pre >= threshold:
                pre_list.append('yes')
            else:
                pre_list.append('no')
            
        precision = 0
        for i in range(len(pre_list)):
            if pre_list[i] == label[0][i]:
                precision = precision + 1
        precision = precision / len(pre_list)
        # print('threshold is :', threshold, 'precison is : ', precision)
        TP = 0
        FP = 0
        FN = 0
        for j in range(len(result[0])):
            if (pre_list[j] == 'no') & (label[0][i] == 'no'):
                TP = TP + 1
            elif (pre_list[j] == 'no') & (label[0][i] == 'yes'):
                FP = FP + 1
            elif (pre_list[j] == 'yes') & (label[0][i] == 'no'):
                FN = FN + 1
                
                
        TP_list.append(TP)
        FP_list.append(FP)
        FN_list.append(FN)
        print('TP_list: ', TP_list)
        print('FP_list: ', FP_list)
        print('FN_list: ', FN_list)
        

        