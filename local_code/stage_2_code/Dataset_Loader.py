'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None

    dataset_source_file_name_train = None
    dataset_source_file_name_test = None

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
        if self.dataset_source_file_name_train and self.dataset_source_file_name_test:
            return {
                'train': self.load(self.dataset_source_file_name_train),
                'test': self.load(self.dataset_source_file_name_test),
            }
        return self.load(self.dataset_source_file_name)