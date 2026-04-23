'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class Setting_Train_Test_Split(setting):
    fold = 3
    
    def load_run_save_evaluate(self):
        
        # load dataset
        # loaded_data = self.dataset.load()
        self.dataset.dataset_source_file_name = 'train.csv'
        train_data = self.dataset.load()

        self.dataset.dataset_source_file_name = 'test.csv'
        test_data = self.dataset.load() 

        X_train, X_test = np.array(train_data['X']), np.array(test_data['X'])
        y_train, y_test = np.array(train_data['y']), np.array(test_data['y'])
        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), 0.0

        