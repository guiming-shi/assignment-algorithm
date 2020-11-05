from math import log

class decision_tree:
    def __init__(self, \
                 select_data, \
                 select_feature_id):
        super(decision_tree, self).__init__()
        self.feature_list = { \
                             'Age': [1, 2, 3, 4, 5, 6, 7, 8, 9], \
                             'Job': ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'], \
                             'Marital': ['divorced', 'married', 'single'], \
                             'Education': ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree'], \
                             'Default': ['no', 'yes'], \
                             'Housing': ['no', 'yes'], \
                             'Loan': ['no', 'yes'], \
                             'Contact': ['cellular','telephone'], \
                             'Month': ['jan', 'feb', 'mar', 'apr' , 'may' , 'jun' , 'jul' , 'aug' , 'sep' , 'oct' , 'nov', 'dec'], \
                             'Dayofweek': ['mon','tue','wed','thu','fri'], \
                             'Campaign': [1, 2, 3, 4, 5], \
                             'Pdays': [1, 2, 3, 4, 5], \
                             'Previous': [1, 2, 3], \
                             'Poutcome': ['failure', 'nonexistent', 'success'], \
                             'Emprate': [1, 2, 3, 4, 5, 6], \
                             'Consprice': [1, 2, 3, 4, 5, 6, 7], \
                             'Conscpnf': [1, 2, 3, 4, 5, 6], \
                             'Euribor3m': [1, 2, 3], \
                             'Nremployed': [1, 2, 3, 4], \
                             'label': ['yes', 'no'] \
        }
        self.select_data = select_data
        self.select_feature_id = select_feature_id
        self.select_feature_number = len(select_data[0]) - 1
        self.select_feature = {}
        self.label_yes_id = []

    def select_data_handle(self):
        for i in self.select_feature_id:
            self.select_feature[i] = set()
        for i in range(len(self.select_data)):
            j = 0
            for key in self.select_feature:
                self.select_feature[key].add(self.select_data[i][j])
                j = j +1



    def id3(self):
        for i in range(len(self.select_data)):
            if self.select_data[i][self.select_feature_number] == 'yes':
                self.label_yes_id.append(i)

        for i in len(self.select_data):


        for i in self.label_yes_id:
