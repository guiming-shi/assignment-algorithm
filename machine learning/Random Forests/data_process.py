# label:
# 0.	Age (numeric) 17-29 30-32 33-35 36-38 39-41 42-46 47-52 53-60 61-
# 1.	Job : type of job (categorical: 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown')
# 2.	Marital : marital status (categorical: 'divorced', 'married', 'single', 'unknown' ; note: 'divorced' means divorced or widowed)
# 3.	Education (categorical: 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown')
# 4.	Default: has credit in default? (categorical: 'no', 'yes', 'unknown')
# 5.	Housing: has housing loan? (categorical: 'no', 'yes', 'unknown')
# 6.	Loan: has personal loan? (categorical: 'no', 'yes', 'unknown')

# 7.	Contact: contact communication type (categorical: 'cellular','telephone')
# 8.	Month: last contact month of year (categorical: 'jan', 'feb', 'mar', 'apr' , 'may' , 'jun' , 'jul' , 'aug' , 'sep' , 'oct' , 'nov', 'dec')
# 9.	Dayofweek: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
# 10.	Duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a reabank_listic predictive model.

# 11.	Campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact) 1 2 3 4-10 11-
# 12.	Pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted) 1-3 4-6 6-12 12-27 999
# 13.	Previous: number of contacts performed before this campaign and for this client (numeric) 0 1 2-
# 14.	Poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success') 

# 15.	Emp.var.rate: employment variation rate - quarterly indicator (numeric) <=-3 -3~-2 -2~-1 -1~0 0~1.2 1.2-
# 16.	Cons.price.idx: consumer price index - monthly indicator (numeric) 92.8 93 93.4 93.7 93.9 94.2 +
# 17.	Cons.conf.idx: consumer confidence index - monthly indicator (numeric) -47 -46 -42 -41 -36 +
# 18.	Euribor3m: euribor 3 month rate - daily indicator (numeric) 1.5 4.8 +
# 19.	Nr.employed: number of employees - quarterly indicator (numeric)  5099 5190  5200 +
# 20.   yes/no


import pandas as pd
import random

class data_process:
    def __init__(self, \
                 bank_list, \
                 Age_split=[29, 32, 35, 38, 41, 46, 52, 60], \
                 Campaign_split=[1, 2, 3, 10], \
                 Pdays_split=[3, 6, 12, 27],\
                 Previous_split=[0, 1], \
                 Emprate_split=[-3, -2, -1, 0, 1.2], \
                 Consprice_split=[92.8, 93, 93.4, 93.7, 93.9, 94.2], \
                 Consconf_split=[-47, -46, -42, -41, -36], \
                 Euribor3m_split=[1.5, 4.8], \
                 Nremployed_split=[5099, 5190, 5200]):
        super(data_process, self).__init__()
        #split number
        self.split = {'Age': [0, Age_split], \
                      'Campaign': [11, Campaign_split], \
                      'Pdays': [12, Pdays_split], \
                      'Previous': [13, Previous_split], \
                      'Emprate': [15, Emprate_split], \
                      'Consprice': [16, Consprice_split], \
                      'Conscpnf': [17, Consconf_split], \
                      'Euribor3m': [18, Euribor3m_split], \
                      'Nremployed': [19, Nremployed_split]
        }
        #del the duration 
        self.split_del = 10
        #handle the unknown type
        self.unknown_label = {'Job': [1, ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'], \
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], \
                        'Marital': [2, ['divorced', 'married', 'single'], \
                                   [0, 0, 0], [0, 0, 0]], \
                        'Education': [3, ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree'], \
                                     [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], \
                        'Default': [4, ['no', 'yes'], \
                                   [0, 0], [0, 0]], \
                        'Housing': [5, ['no', 'yes'], \
                                   [0, 0], [0, 0]], \
                        'Loan': [6, ['no', 'yes'], \
                                [0, 0], [0, 0]]
        }
        self.unknown_string = 'unknown'
        # the number of yes and no label
        self.number_yes = 0
        self.number_no = 0
        # the self.bank_list_buffer is the output
        self.bank_list = bank_list
        self.bank_list_buffer = [None] * len(self.bank_list)
        # handle the continuous variables, split from the list self.split, from 1 to N
        self.embedding()
        # handle the unknown type, according to the probability of label of same output type 
        self.missing_process()
        # del the duration feature
        self.del_Duration()
        # sort the data according the output type
        self.class_sort()
        self.change_idx = [] 
        self.testVec = []
        
    def embedding(self):
        for i in range(len(self.bank_list)):
            for key in self.split:
                for j in range(len(self.split[key][1]) + 1):
                    if j < len(self.split[key][1]):
                        if self.bank_list[i][self.split[key][0]] <= self.split[key][1][j]:
                            self.bank_list[i][self.split[key][0]] = j + 1
                            break
                    else:
                        self.bank_list[i][self.split[key][0]] = j + 1
                        break

    #missing 1st to 6th data process 
    def missing_process(self):
        for i in range(len(self.bank_list)):
            for key in self.unknown_label:
                for j in range(len(self.unknown_label[key][1])):
                    if self.bank_list[i][self.unknown_label[key][0]] == self.unknown_label[key][1][j]:
                        if self.bank_list[i][20] == 'yes':
                            self.unknown_label[key][2][j] = self.unknown_label[key][2][j] + 1
                        elif self.bank_list[i][20] == 'no':
                            self.unknown_label[key][3][j] = self.unknown_label[key][3][j] + 1

        for i in range(len(self.bank_list)):
            for key in self.unknown_label:
                missing_sum = sum(self.unknown_label[key][2])
                for j in range(len(self.unknown_label[key][2])):
                    self.unknown_label[key][2][j] = self.unknown_label[key][2][j] /  missing_sum
                missing_sum = sum(self.unknown_label[key][3])
                for j in range(len(self.unknown_label[key][3])):
                    self.unknown_label[key][3][j] = self.unknown_label[key][3][j] /  missing_sum

        for key in self.unknown_label:
            for j in range(len(self.unknown_label[key][2])):
                if j >= 1:
                    self.unknown_label[key][2][j] = self.unknown_label[key][2][j] + self.unknown_label[key][2][j-1]
            for j in range(len(self.unknown_label[key][3])):
                if j >= 1:
                    self.unknown_label[key][3][j] = self.unknown_label[key][3][j] + self.unknown_label[key][3][j-1]
            
        for i in range(len(self.bank_list)):
            if self.bank_list[i][20] == 'yes':
                for key in self.unknown_label:
                    if self.bank_list[i][self.unknown_label[key][0]] == self.unknown_string:
                        randint_number = random.random()
                        for j in range(len(self.unknown_label[key][2])):
                            if randint_number <= self.unknown_label[key][2][j]:
                                self.bank_list[i][self.unknown_label[key][0]] = self.unknown_label[key][1][j]
            elif self.bank_list[i][20] == 'no':
                for key in self.unknown_label:
                    if self.bank_list[i][self.unknown_label[key][0]] == self.unknown_string:
                        randint_number = random.random()
                        for j in range(len(self.unknown_label[key][3])):
                            if randint_number <= self.unknown_label[key][3][j]:
                                self.bank_list[i][self.unknown_label[key][0]] = self.unknown_label[key][1][j]

    def del_Duration(self):
        for i in range(len(self.bank_list)):
            del(self.bank_list[i][self.split_del])

#the index of yes/no from 20 to 19 is because the self.del_Duration
    def class_sort(self):
        j = 0
        k = 1
        for i in range(len(self.bank_list)):
            if self.bank_list[i][19] == 'yes':
                self.number_yes = self.number_yes + 1
                self.bank_list_buffer[j] = self.bank_list[i]
                j = j + 1
            elif self.bank_list[i][19] == 'no':
                self.number_no = self.number_no + 1
                self.bank_list_buffer[len(self.bank_list_buffer) - k] = self.bank_list[i]
                k = k + 1
    
    def select(self, select_number, feature_number):
        index = random.sample(range(0, 18), feature_number)
        index = sorted(index)
        select_bank_data = [[] for i in range(select_number)]
        select_bank_data_id = set()
        for i in range(select_number):
            select_split = self.number_yes / (self.number_yes + self.number_no)
            select_split_randint = random.random()
            #if select_split_randint > select_split:
            if select_split_randint > 0.5:   
                select_data = random.sample(range(0, self.number_yes), 1)[0]
                select_bank_data_id.add(select_data)
                for j in index:
                    select_bank_data[i].append(self.bank_list_buffer[select_data][j])
                select_bank_data[i].append(self.bank_list_buffer[select_data][19])
            else:
                select_data = random.sample(range(self.number_yes, self.number_yes + self.number_no), 1)[0]
                select_bank_data_id.add(select_data)
                for j in index:
                    select_bank_data[i].append(self.bank_list_buffer[select_data][j])
                select_bank_data[i].append(self.bank_list_buffer[select_data][19])

        index.append(-1)

        test_data = []
        test_data_idx = []
        for i in range(len(self.bank_list)):
            if i not in select_bank_data_id:
                test_data.append([self.bank_list[i][k] for k in index])
                test_data_idx.append(i)
        
        return select_bank_data, select_bank_data_id , index[:-1] , test_data, test_data_idx

    def get_test_data(self, feature_idx):
        train_data = []
        for i in range(len(self.bank_list)):
            train_data.append([self.bank_list[i][k] for k in feature_idx])
        label = []
        for i in range(len(self.bank_list)):
            label.append(self.bank_list[i][-1])
        return train_data

    def get_test_label(self):
        label = []
        for i in range(len(self.bank_list)):
            label.append(self.bank_list[i][-1])
        return label

    def get_unknown_label(self):
        return self.unknown_label

    def get_bank_list(self):
        return self.bank_list_buffer

class data_select:
    def __init__(self,\
                 bank_list):
        super(data_select, self).__init__()


if __name__ == '__main__':
    data = pd.read_csv("bank-additional-full.csv")
    bank_list = data.values.tolist()
    data_processor = data_process(bank_list=bank_list)
    bank_list = data_processor.get_bank_list()
    select_data, select_data_id, select_feature_id, test_data = data_processor.select(2000, 5)