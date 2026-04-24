'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        # return accuracy_score(self.data['true_y'], self.data['pred_y'])
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']
        return {
            'Accuracy': accuracy_score(true_y, pred_y),
            'F1 weighted': f1_score(true_y, pred_y, average='weighted', zero_division=0),
            'F1 macro' : f1_score(true_y, pred_y, average='macro', zero_division=0),
            'F1 micro' : f1_score(true_y, pred_y, average='micro', zero_division=0),
            'Precision weighted': precision_score(true_y, pred_y, average='weighted', zero_division=0),
            'Precision macro': precision_score(true_y, pred_y, average='macro', zero_division=0),
            'Precision micro': precision_score(true_y, pred_y, average='micro', zero_division=0),
            'Recall weighted': recall_score(true_y, pred_y, average='weighted', zero_division=0),
            'Recall macro': recall_score(true_y, pred_y, average='macro', zero_division=0),
            'Recall micro': recall_score(true_y, pred_y, average='micro', zero_division=0)
        }
        