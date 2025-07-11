import torch
import torch.nn as nn
from torchinfo import summary

class ConvBlock(nn.Module):
    def __init__(self, kernel_size, dim_in, dim, dim_out):
        super().__init__()
        self.kernelsize = kernel_size
        self.dim = dim
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj1 = nn.Conv1d(dim_in, dim, self.kernelsize, padding='same')
        self.proj2 = nn.Conv1d(dim, dim_out, self.kernelsize, padding='same')
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward(self, x,):
        x = self.proj1(x)            
        x = self.act1(x)
        x = self.proj2(x)            
        x = self.act2(x)
        return x

class cnn(nn.Module):
    def __init__(self, kernel_size = 3, dropout = 0.4, n_fc_neurons = 128, n_filters = [48, 96, 96, 192, 192], final_act='sigmoid'):
        super().__init__()
        self.channels = 2
        self.kernelsize = kernel_size
        self.dropout = dropout
        self.n_fc_neurons = n_fc_neurons
        self.n_filters = n_filters

        self.cb1 = ConvBlock(self.kernelsize, self.channels, self.n_filters[0], self.n_filters[0])
        self.cb2 = ConvBlock(self.kernelsize, self.n_filters[0], self.n_filters[1], self.n_filters[1])
        self.cb3 = ConvBlock(self.kernelsize, self.n_filters[1], self.n_filters[2], self.n_filters[2])
        self.conv5a = nn.Conv1d(n_filters[2],n_filters[4],kernel_size,padding='same')
        self.conv5b = nn.Conv1d(n_filters[4],n_filters[4],kernel_size,padding='same')
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(self.kernelsize)

        ### FC part 
        self.dense1 = nn.Linear(192,n_fc_neurons)
        self.drop1 = nn.Dropout(p=dropout)
        self.dense2 = nn.Linear(n_fc_neurons, n_fc_neurons)
        self.drop2 = nn.Dropout(dropout)
        self.denseout = nn.Linear(n_fc_neurons, 1)
        if final_act == 'relu':
            self.final_act = nn.ReLU()
        elif final_act == 'sigmoid':
            self.final_act = nn.Sigmoid()
        elif final_act == 'tanh':
            self.final_act = nn.Tanh()
        else:
            raise ValueError("final_act must be one of ['relu', 'sigmoid', 'tanh']")

    
    def forward(self, x):
        x = self.cb1(x)
        x = self.pool(x)
        x = self.cb2(x)
        x = self.pool(x)
        x = self.cb3(x)
        x = self.pool(x)
        
        x = self.act(self.conv5a(x))
        x = self.act(self.conv5b(x))
        x = torch.mean(x,dim=-1)
        
        ### FC part
        x = self.act(self.dense1(x))
        x = self.drop1(x)
        x = self.act(self.dense2(x))
        x = self.drop2(x)
        x = self.final_act(self.denseout(x))

        return x
    
    def train_step(self, x, y, optimizer, loss='bce'):

        # zero grad
        optimizer.zero_grad()
        pred = self.forward(x)
        if loss == 'bce':
            # binary cross entropy loss
            loss = nn.BCELoss()(pred.squeeze(), y.float())
        elif loss == 'mse':
            # mean squared error loss
            loss = nn.MSELoss()(pred.squeeze(), y.float())
        # backward pass
        loss.backward()
        # update weights
        optimizer.step()
        return loss.item(), pred.squeeze()
    
    def test_step(self, x, y, loss='bce'):
        with torch.no_grad():
            pred = self.forward(x)
            if loss == 'bce':
                # binary cross entropy loss
                loss = nn.BCELoss()(pred.squeeze(), y.float())
            elif loss == 'mse':
                # mean squared error loss
                loss = nn.MSELoss()(pred.squeeze(), y.float())
        return loss.item(), pred.squeeze()



if __name__ == "__main__":
    # Example usage
    model = cnn()
    summary(model, input_size=(1, 2, 180))  # Example input size (batch_size, channels, sequence_length)