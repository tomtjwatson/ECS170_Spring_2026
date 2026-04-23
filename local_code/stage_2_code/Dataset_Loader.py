'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self, file_name):
        X = []
        y = []
        file = open(self.dataset_source_folder_path + file_name, 'r')
        for line in file:
            elements = [int(i) for i in line.strip('\n').split(',')]
            y.append(elements[0])
            X.append(elements[1:])
        return {'X': X, 'y': y}

    def load_data(self):
        print('loading data...')
        return {
            'train': self.load('train.csv'),
            'test':  self.load('test.csv'),
        }