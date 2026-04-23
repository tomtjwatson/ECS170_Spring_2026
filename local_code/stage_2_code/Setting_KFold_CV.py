'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from sklearn.model_selection import KFold
import numpy as np

class Setting_KFold_CV(setting):
    fold = 3
    test_dataset = None
    
    def load_run_save_evaluate(self):
        
        # load dataset
        self.dataset.dataset_source_file_name = 'train.csv'
        train_data = self.dataset.load()

        self.test_dataset.dataset_source_file_name = 'test.csv'
        test_data = self.test_dataset.load()

        X_train_full = np.array(train_data['X'])
        y_train_full = np.array(train_data['y'])
        X_test_final = np.array(test_data['X'])
        y_test_final = np.array(test_data['y'])
        
        kf = KFold(n_splits=self.fold, shuffle=True)
        
        fold_count = 0
        score_list = []
        for train_index, test_index in kf.split(X_train_full):
            fold_count += 1
            print('************ Fold:', fold_count, '************')
            X_train_fold = X_train_full[train_index]   
            y_train_fold = y_train_full[train_index]
        
            # run MethodModule
            self.method.data = {'train': {'X': X_train_fold, 'y': y_train_fold}, 'test': {'X': X_test_final, 'y': y_test_final}}
            learned_result = self.method.run()
            
            # save raw ResultModule
            self.result.data = learned_result
            self.result.fold_count = fold_count
            self.result.save()
            
            self.evaluate.data = learned_result
            score_list.append(self.evaluate.evaluate())
        
        return np.mean(score_list), np.std(score_list)

        