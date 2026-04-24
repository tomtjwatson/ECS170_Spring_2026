'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.stage_2_code.Method_MLP import Method_MLP
from torch import nn

### ADDED FEAT: 3 layers + normalize batches
class Method_MLP_exp1(Method_MLP):

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        super().__init__(mName, mDescription)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(784, 256)
        self.bn_1 = nn.BatchNorm1d(256)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(256, 128)
        self.bn_2 = nn.BatchNorm1d(128)
        self.activation_func_2 = nn.ReLU()
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.fc_layer_3 = nn.Linear(128, 10)
        self.activation_func_3 = nn.Softmax(dim=1)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer
    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h = self.activation_func_1(self.bn_1(self.fc_layer_1(x)))
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        h = self.activation_func_2(self.bn_2(self.fc_layer_2(h)))
        y_pred = self.activation_func_3(self.fc_layer_3(h))
        return y_pred
