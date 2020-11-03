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


import pandas as pd
import random

class data_process:
    def __init__(self,\
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
        # self.split_number = [0,11,12,13,15,16,17,18,19]
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
        self.split_del = 10
        
        self.unknown_label = {'Job': [1, ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed']], \
                        'Marital': [2, ['divorced', 'married', 'single']], \
                        'Education': [3, ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree']], \
                        'Default': [4, ['no', 'yes']], \
                        'Housing': [5, ['no', 'yes']], \
                        'Loan': [6, ['no', 'yes']]
        }
        self.unknown_string = 'unknown'
        self.missing_count = {'Job': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                            'Marital': [0, 0, 0], \
                            'Education': [0, 0, 0, 0, 0, 0, 0], \
                            'Default': [0, 0], \
                            'Housing': [0, 0], \
                            'Loan': [0, 0]}

        self.bank_list = bank_list
        self.embedding()
        self.missing_process()
        self.del_Duration()
        
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
                for j in range(len(self.missing_count[key])):
                    if self.bank_list[i][self.unknown_label[key][0]] == self.unknown_label[key][1][j]:
                        self.missing_count[key][j] = self.missing_count[key][j] + 1
        for i in range(len(self.bank_list)):
            for key in self.missing_count:
                missing_sum = sum(self.missing_count[key])
                for j in range(len(self.missing_count[key])):
                    self.missing_count[key][j] = self.missing_count[key][j] /  missing_sum
        for key in self.missing_count:
            for i in range(len(self.missing_count[key])):
                if i >= 1:
                    self.missing_count[key][i] = self.missing_count[key][i] + self.missing_count[key][i-1]
        for i in range(len(self.bank_list)):
            for key in self.unknown_label:
                if self.bank_list[i][self.unknown_label[key][0]] == self.unknown_string:
                    randint_number = random.randint(0, 1)
                    for j in range(len(self.missing_count[key])):
                        if randint_number <= self.missing_count[key][j]:
                            self.bank_list[i][self.unknown_label[key][0]] = self.unknown_label[key][1][j]


    def get_missing_count(self):
        return self.missing_count

             
    def del_Duration(self):
        for i in range(len(self.bank_list)):
            del(self.bank_list[i][self.split_del])




    def get_bank_list(self):
        return self.bank_list

class data_select:
    def __init__(self,\
                 bank_list):
        super(data_select, self).__init__()


data = pd.read_csv("bank-additional-full.csv")
bank_list = data.values.tolist()
data_processor = data_process(bank_list=bank_list)
bank_list = data_processor.get_bank_list()