'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 100
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(784, 256)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(256, 10)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.activation_func_2 = nn.Softmax(dim=1)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h = self.activation_func_1(self.fc_layer_1(x))
        # outout layer result
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        y_pred = self.activation_func_2(self.fc_layer_2(h))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        loss_history = []
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            loss_history.append(train_loss.item())

            if epoch%10 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                metrics = accuracy_evaluator.evaluate()
                print(f'Epoch: {epoch} | Accuracy: {metrics["Accuracy"]}, '
                      f'Loss: {train_loss.item()}, F1(w): {metrics["F1 weighted"]}, '
                      f'Precision(w): {metrics["Precision weighted"]}, Recall(w): {metrics["Recall weighted"]}\n')

        plt.figure()
        plt.plot(range(self.max_epoch), loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Epoch vs. Loss Plot')
        plt.savefig('../../script/stage_2_script/MLP_convergence_plot.png')
        print(f"MLP_convergence_plot.png created")
    
    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]
    
    def evaluate_metrics(self, true_y, pred_y):
        accuracy  = accuracy_score(true_y, pred_y)

        f1_weighted = f1_score(true_y, pred_y, average='weighted', zero_division=0)
        f1_macro = f1_score(true_y, pred_y, average='macro', zero_division=0)
        f1_micro = f1_score(true_y, pred_y, average='micro', zero_division=0)

        recall_weighted = recall_score(true_y, pred_y, average='weighted', zero_division=0)
        recall_macro = recall_score(true_y, pred_y, average='macro', zero_division=0)
        recall_micro = recall_score(true_y, pred_y, average='micro', zero_division=0)

        precision_weighted = precision_score(true_y, pred_y, average='weighted', zero_division=0)
        precision_macro = precision_score(true_y, pred_y, average='macro', zero_division=0)
        precision_micro = precision_score(true_y, pred_y, average='micro', zero_division=0)

        print(f'  Accuracy:            {accuracy:.4f}')
        print(f'  F1(weighted):       {f1_weighted:.4f}')
        print(f'  F1(macro):          {f1_macro:.4f}')
        print(f'  F1(micro):          {f1_micro:.4f}')
        print(f'  Recall(weighted):   {recall_weighted:.4f}')
        print(f'  Recall(macro):      {recall_macro:.4f}')
        print(f'  Recall(micro):      {recall_micro:.4f}')
        print(f'  Precision(weighted):{precision_weighted:.4f}')
        print(f'  Precision(macro):   {precision_macro:.4f}')
        print(f'  Precision(micro):   {precision_micro:.4f}')

        return {
            'accuracy':           accuracy,
            'f1_weighted':        f1_weighted,
            'f1_macro':           f1_macro,
            'f1_micro':           f1_micro,
            'recall_weighted':    recall_weighted,
            'recall_macro':       recall_macro,
            'recall_micro':       recall_micro,
            'precision_weighted': precision_weighted,
            'precision_macro':    precision_macro,
            'precision_micro':    precision_micro,
        }
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        true_y = self.data['test']['y']
        metrics = self.evaluate_metrics(true_y, pred_y)
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], 'metrics': metrics}
            