'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.stage_2_code.Method_MLP import Method_MLP
from torch import nn

### EXPERIMENTING: layers and norm full batch
class Method_MLP_exp2(Method_MLP):

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        super().__init__(mName, mDescription)

        # ---- Layer configuration: edit this list to add or remove hidden layers ----
        # Format: [input_size, hidden_1, hidden_2, ..., output_size]
        # The last size is always the output (10 classes for MNIST)
        layer_sizes = [784, 1024, 512, 256, 128, 64, 10]
        # ----------------------------------------------------------------------------

        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        # Build hidden layers (everything except the final output layer)
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(len(layer_sizes) - 2): # -2 to ignore going to output layer
            self.hidden_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i + 1]))

        self.activation_func = nn.ReLU()

        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.fc_output = nn.Linear(layer_sizes[-2], layer_sizes[-1]) # x --> 10
        self.activation_func_out = nn.Softmax(dim=1)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer
    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h = x
        for i in range(len(self.hidden_layers)):
            h = self.activation_func(self.batch_norms[i](self.hidden_layers[i](h)))
        # self.fc_output(h) will be a nx10 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        y_pred = self.activation_func_out(self.fc_output(h))
        return y_pred
