import torch
import torch.nn as nn
import torch.nn.functional as F

window = 180

kernel_size = 1

dropout = 0.4

n_fc_neurons = 64

n_filters = [24, 48, 48, 96, 192]

#batchsize = 10000

class cnn_class(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = nn.Conv1d(window,n_filters[0],kernel_size,padding='same')
        #self.conv1b = nn.Conv1d(window,n_filters[1],kernel_size,padding='same')
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size)
        
        self.conv2 = nn.Conv1d(n_filters[0],n_filters[1],kernel_size,padding='same')
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size)
        self.conv3 = nn.Conv1d(n_filters[1],n_filters[2],kernel_size,padding='same')
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size)
        self.conv4 = nn.Conv1d(n_filters[2],n_filters[3],kernel_size,padding='same')
        self.act4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size)

        self.conv5 = nn.Conv1d(n_filters[3],n_filters[4],kernel_size,padding='same')
        self.act5 = nn.ReLU()
        self.pool5 = nn.AvgPool1d(kernel_size) #is it global?

        ### FC part 
        self.dense1 = nn.Linear(2,n_fc_neurons)
        self.drop1 = nn.Dropout(p=dropout)
        self.dense2 = nn.Linear(n_fc_neurons, n_fc_neurons)
        self.drop2 = nn.Dropout(dropout)
        self.denseOut = nn.Linear(n_fc_neurons, 1)

    
    def forward(self, x):
        x = self.act1(self.conv1a(x))
        print("conv1")
        #x = self.act1(self.conv1b(x))
        x = self.pool1(x)
        print("pool1 finished")
        x = self.act2(self.conv2(x))
        #x = self.act2(self.conv2(x))
        x = self.pool2(x)
        print("pool2 finished")
        x = self.act3(self.conv3(x))
        #x = self.act3(self.conv3(x))
        x = self.pool3(x)
        print("pool3 finished")
        x = self.act4(self.conv4(x))
        #x = self.act4(self.conv4(x))
        x = self.pool4(x)
        print("pool4 finished")
        x = self.act5(self.conv5(x))
        #x = self.act5(self.conv5(x))
        x = self.pool5(x)
        print("pool5 finished")
        
        ### FC part
        x = nn.ReLU()(self.dense1(x))
        print("ReLU1 finished")
        x = self.drop1(x)
        print("first drop")
        x = nn.ReLU()(self.dense2(x))
        print("ReLU2 finished")
        x = self.drop2(x)
        print("second drop")
        x = nn.Sigmoid()(self.denseOut(x))

        return x 


        
