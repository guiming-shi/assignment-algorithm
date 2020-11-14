from math import log2
# select_data[0] is the data
# select_data[1] is the index of the data
# select_data[1] is the index of the feature
# len(select_data[0][0]) is the number of the feature(add the label(yes/no))
class decision_tree:
    def __init__(self, \
                 select_data):
        super(decision_tree, self).__init__()
        self.feature_list = { \
                             'Age': {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0], 8: [0, 0], 9: [0, 0]}, \
                             'Job': {'admin.': [0, 0], 'blue-collar': [0, 0], 'entrepreneur': [0, 0], 'housemaid': [0, 0], 'management': [0, 0], 'retired': [0, 0], 'self-employed': [0, 0], \
                                     'services': [0, 0], 'student': [0, 0], 'technician': [0, 0], 'unemployed': [0, 0]}, \
                             'Marital': {'divorced': [0, 0], 'married': [0, 0], 'single': [0, 0]}, \
                             'Education': {'basic.4y': [0, 0], 'basic.6y': [0, 0], 'basic.9y': [0, 0], 'high.school': [0, 0], 'illiterate': [0, 0], 'professional.course': [0, 0], 'university.degree': [0, 0]}, \
                             'Default': {'no': [0, 0], 'yes': [0, 0]}, \
                             'Housing': {'no': [0, 0], 'yes': [0, 0]}, \
                             'Loan': {'no': [0, 0], 'yes': [0, 0]}, \
                             'Contact': {'cellular': [0, 0],'telephone': [0, 0]}, \
                             'Month': {'jan': [0, 0], 'feb': [0, 0], 'mar': [0, 0], 'apr': [0, 0], 'may': [0, 0], 'jun': [0, 0], 'jul': [0, 0], 'aug': [0, 0], 'sep': [0, 0], 'oct': [0, 0], 'nov': [0, 0], 'dec': [0, 0]}, \
                             'Dayofweek': {'mon': [0, 0], 'tue': [0, 0], 'wed': [0, 0], 'thu': [0, 0], 'fri': [0, 0]}, \
                             'Campaign': {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0]}, \
                             'Pdays': {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0]}, \
                             'Previous': {1: [0, 0], 2: [0, 0], 3: [0, 0]}, \
                             'Poutcome': {'failure': [0, 0], 'nonexistent': [0, 0], 'success': [0, 0]}, \
                             'Emprate': {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0]}, \
                             'Consprice': {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0]}, \
                             'Conscpnf': {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0]}, \
                             'Euribor3m': {1: [0, 0], 2: [0, 0], 3: [0, 0]}, \
                             'Nremployed': {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]}, \
                             'label': {'yes': [0, 0], 'no': [0, 0]} \
        }
        # print(self.feature_list)
        #the first one is the yes number and the last one is the no number
        self.feature = ['Age', 'Job', 'Marital', 'Education', 'Default', 'Housing', 'Loan', 'Contact', 'Month', 'Dayofweek', 'Campaign', 'Pdays', 'Previous', 'Poutcome', 'Emprate', \
                        'Consprice', 'Conscpnf', 'Euribor3m', 'Nremployed', 'label']
        self.feature_detail = { \
                             'Age': {1, 2, 3, 4, 5, 6, 7, 8, 9}, \
                             'Job': {'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', \
                                     'services', 'student', 'technician', 'unemployed'}, \
                             'Marital': {'divorced', 'married', 'single'}, \
                             'Education': {'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree'}, \
                             'Default': {'no', 'yes'}, \
                             'Housing': {'no', 'yes'}, \
                             'Loan': {'no', 'yes'}, \
                             'Contact': {'cellular','telephone'}, \
                             'Month': {'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'}, \
                             'Dayofweek': {'mon', 'tue', 'wed', 'thu', 'fri'}, \
                             'Campaign': {1, 2, 3, 4, 5}, \
                             'Pdays': {1, 2, 3, 4, 5}, \
                             'Previous': {1, 2, 3}, \
                             'Poutcome': {'failure', 'nonexistent', 'success'}, \
                             'Emprate': {1, 2, 3, 4, 5, 6}, \
                             'Consprice': {1, 2, 3, 4, 5, 6, 7}, \
                             'Conscpnf': {1, 2, 3, 4, 5, 6}, \
                             'Euribor3m': {1, 2, 3}, \
                             'Nremployed': {1, 2, 3, 4}, \
                             'label': {'yes', 'no'} \
        }
        self.select_data = select_data
        self.select_feature_number = len(self.select_data[0][0]) - 1
        self.HD_id3 = [0] * len(self.select_data[2])
        self.id_tree = {}
        self.tree_id_x = 0

        self.tree1()
        # self.tree2()

    def feature_list_clear(self):
        self.feature_list = { \
                             'Age': {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0], 8: [0, 0], 9: [0, 0]}, \
                             'Job': {'admin.': [0, 0], 'blue-collar': [0, 0], 'entrepreneur': [0, 0], 'housemaid': [0, 0], 'management': [0, 0], 'retired': [0, 0], 'self-employed': [0, 0], \
                                     'services': [0, 0], 'student': [0, 0], 'technician': [0, 0], 'unemployed': [0, 0]}, \
                             'Marital': {'divorced': [0, 0], 'married': [0, 0], 'single': [0, 0]}, \
                             'Education': {'basic.4y': [0, 0], 'basic.6y': [0, 0], 'basic.9y': [0, 0], 'high.school': [0, 0], 'illiterate': [0, 0], 'professional.course': [0, 0], 'university.degree': [0, 0]}, \
                             'Default': {'no': [0, 0], 'yes': [0, 0]}, \
                             'Housing': {'no': [0, 0], 'yes': [0, 0]}, \
                             'Loan': {'no': [0, 0], 'yes': [0, 0]}, \
                             'Contact': {'cellular': [0, 0],'telephone': [0, 0]}, \
                             'Month': {'jan': [0, 0], 'feb': [0, 0], 'mar': [0, 0], 'apr': [0, 0], 'may': [0, 0], 'jun': [0, 0], 'jul': [0, 0], 'aug': [0, 0], 'sep': [0, 0], 'oct': [0, 0], 'nov': [0, 0], 'dec': [0, 0]}, \
                             'Dayofweek': {'mon': [0, 0], 'tue': [0, 0], 'wed': [0, 0], 'thu': [0, 0], 'fri': [0, 0]}, \
                             'Campaign': {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0]}, \
                             'Pdays': {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0]}, \
                             'Previous': {1: [0, 0], 2: [0, 0], 3: [0, 0]}, \
                             'Poutcome': {'failure': [0, 0], 'nonexistent': [0, 0], 'success': [0, 0]}, \
                             'Emprate': {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0]}, \
                             'Consprice': {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0]}, \
                             'Conscpnf': {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0]}, \
                             'Euribor3m': {1: [0, 0], 2: [0, 0], 3: [0, 0]}, \
                             'Nremployed': {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]}, \
                             'label': {'yes': [0, 0], 'no': [0, 0]} \
        }

    def tree1(self):
        self.feature_list.clear()
        self.feature_list_clear()

        for id_x in range(self.select_feature_number):
            self.select_feature_class1(id_x = id_x)
        
        for id_x in range(self.select_feature_number):
            self.id31(id_x = id_x)

        # self.feature_list_clear()

        self.tree_id_x = [0] * 5

        HD = self.HD_id3[0]
        for i in range(len(self.HD_id3)):
            if self.HD_id3[i] < HD:
                HD = self.HD_id3[i]
                self.tree_id_x[0] = i
        
        self.id_tree[self.feature[self.tree_id_x[0]]] = {}

        for i in self.feature_detail[self.feature[self.tree_id_x[0]]]:
            self.id_tree[self.feature[self.tree_id_x[0]]][i] = {'yes' if self.feature_list[self.feature[self.tree_id_x[0]]][i][0] > self.feature_list[self.feature[self.tree_id_x[0]]][i][1] else 'no'}

    def select_feature_class1(self, id_x):
        for data in self.select_data[0]:
            if data[len(self.select_data[0][0]) - 1] == 'yes':
                self.feature_list[ self.feature[ self.select_data[2][id_x] ] ] [data[id_x]] [0] = self.feature_list[ self.feature[ self.select_data[2][id_x] ] ] [data[id_x]] [0] + 1
            elif data[len(self.select_data[0][0]) - 1] == 'no':
                self.feature_list[ self.feature[ self.select_data[2][id_x] ] ] [data[id_x]] [1] = self.feature_list[ self.feature[ self.select_data[2][id_x] ] ] [data[id_x]] [1] + 1

    def id31(self, id_x):
        self.HD_id3[id_x] = 0
        for key in self.feature_list[ self.feature[ self.select_data[2][id_x] ] ]:
            yes_number = self.feature_list[ self.feature[ self.select_data[2][id_x] ] ] [ key ] [0]
            no_number = self.feature_list[ self.feature[ self.select_data[2][id_x] ] ] [ key ] [1]
            total_portion = yes_number + no_number
            if total_portion < 1:
                continue
            yes_pro_portion = yes_number / total_portion
            no_pro_portion = no_number / total_portion
            total = len(self.select_data[0])
            proportion = (yes_number + no_number) / total
            self.HD_id3[id_x] = self.HD_id3[id_x] - proportion * ( yes_pro_portion*log2(yes_pro_portion if yes_pro_portion > 0 else 1) + no_pro_portion*log2(no_pro_portion if no_pro_portion > 0 else 1) )
        print(self.HD_id3[id_x])

    # def tree2(self):
    #     for top_tree in self.id_tree:
    #         for top_tree_label in self.id_tree[top_tree]:
    #             self.feature_list.clear()
    #             self.feature_list_clear()
    #             del self.feature_list[top_tree]

    #             for id_x in range(self.select_feature_number):
    #                 if id_x != self.tree_id_x[0]:
    #                     self.select_feature_class2(id_x = id_x)


    # def select_feature_class2(self, id_x):
    #     for data in self.select_data[0]:
    #         if(data[ self.feature [ self.select_data[2][ self.tree_id_x[0] ] ] ]) == self.feature_detail[ self.feature [ self.select_data[2][ self.tree_id_x[0] ] ] ][id_x]:
    #             if data[len(self.select_data[0][0]) - 1] == 'yes':
    #                 self.feature_list[ self.feature[ self.select_data[2][id_x] ] ] [data[id_x]] [0] = self.feature_list[ self.feature[ self.select_data[2][id_x] ] ] [data[id_x]] [0] + 1
    #             elif data[len(self.select_data[0][0]) - 1] == 'no':
    #                 self.feature_list[ self.feature[ self.select_data[2][id_x] ] ] [data[id_x]] [1] = self.feature_list[ self.feature[ self.select_data[2][id_x] ] ] [data[id_x]] [1] + 1


                


            
        