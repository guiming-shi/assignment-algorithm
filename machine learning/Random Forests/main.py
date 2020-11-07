# -*- coding: utf-8 -*-
__author__ = 'guiming'

import data_process as process
import decision_tree_inst as dt
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv("bank-additional-full.csv")
    bank_list = data.values.tolist()
    data_processor = process.data_process(bank_list=bank_list)
    # bank_list = data_processor.get_bank_list()

    data = data_processor.select(10000, 10)
    model = dt.decision_tree(select_data = data)
    
    
    print(model.id_tree)



