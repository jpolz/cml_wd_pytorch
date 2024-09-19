import torch
import torch.nn as nn
import torch.nn.functional as F


class cnn_class(nn.Module):
    def __init__(self, window = 180, kernel_size = 3, dropout = 0.4, n_fc_neurons = 64, n_filters = [24, 48, 48, 96, 192],):
        super().__init__()
        self.channels = 2
        self.kernelsize = kernel_size
        self.dropout = dropout
        self.n_fc_neurons = n_fc_neurons
        self.n_filters = n_filters

        self.conv5a = nn.Conv1d(n_filters[2],n_filters[4],kernel_size,padding='same')
        self.conv5b = nn.Conv1d(n_filters[4],n_filters[4],kernel_size,padding='same')
        self.act5 = nn.ReLU()
        # self.pool5 = nn.AvgPool1d(kernel_size) #is it global?

        ### FC part 
        self.dense1 = nn.Linear(192,n_fc_neurons)
        self.drop1 = nn.Dropout(p=dropout)
        self.dense2 = nn.Linear(n_fc_neurons, n_fc_neurons)
        self.drop2 = nn.Dropout(dropout)
        self.denseOut = nn.Linear(n_fc_neurons, 1)
    
    def inner_layer(self, x, filter_in, filter_out, kernelsize):
        x = nn.ReLU()(nn.Conv1d(filter_in, filter_out, kernelsize, padding='same')(x))
        x = nn.ReLU()(nn.Conv1d(filter_out, filter_out, kernelsize, padding='same')(x))
        return nn.MaxPool1d(kernelsize)(x)

    
    def forward(self, x):
        x = self.inner_layer(x, self.channels, self.n_filters[0], self.kernelsize)
        x = self.inner_layer(x, self.n_filters[0], self.n_filters[1], self.kernelsize)
        x = self.inner_layer(x, self.n_filters[1], self.n_filters[2], self.kernelsize)
        
        x = self.act5(self.conv5a(x))
        x = self.act5(self.conv5b(x))
        x = torch.mean(x,dim=-1)
        
        ### FC part
        x = nn.ReLU()(self.dense1(x))
        x = self.drop1(x)
        x = nn.ReLU()(self.dense2(x))
        x = self.drop2(x)
        x = nn.Sigmoid()(self.denseOut(x))

        return x
