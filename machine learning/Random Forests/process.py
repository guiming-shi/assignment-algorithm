feature_list = ['Age', 'Job', 'Marital', 'Education', 'Default', 'Housing', 'Loan', 'Contact', 'Month', 'Dayofweek', 'Campaign', 'Pdays', 'Previous', 'Poutcome', 'Emprate', \
                        'Consprice', 'Conscpnf', 'Euribor3m', 'Nremployed', 'label']
"""
函数说明:冒泡算法，更改测试集中数据的列排序，以适应决策树
Parameters:
    change_idx, testVec - 更改序列，测试集
Returns:
    change_idx, testVec - 更改后序列，测试集
"""
def BubbleSort(change_idx, testVec):
    n = len(change_idx)
    if n <= 1:
        return change_idx, testVec
    for i in range (0, n):
        for j in range(0, n-i-1):
            if change_idx[j] > change_idx[j+1]:
                (change_idx[j], change_idx[j+1]) = (change_idx[j+1], change_idx[j])
                for k in range(len(testVec)):
                    temp = testVec[k][j]
                    testVec[k][j] = testVec[k][j+1]
                    testVec[k][j+1] = temp
    return change_idx, testVec
"""
函数说明:生成更改序列
Parameters:
    featLabels - 更改序列标签
Returns:
    change_idx -  更改序列
"""
def change_idx(featLabels):
    change_idx = []
    for i in featLabels:
        for feature in feature_list:
            if i == feature:
                change_idx.append(feature_list.index(feature))
    return change_idx