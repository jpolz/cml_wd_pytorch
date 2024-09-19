import torch
import torch.nn as nn
import torch.nn.functional as F


class cnn_class(nn.Module):
    def __init__(self, window = 180, kernel_size = 3, dropout = 0.4, n_fc_neurons = 64, n_filters = [24, 48, 48, 96, 192],):
        super().__init__()
        self.conv1a = nn.Conv1d(window,n_filters[0],kernel_size,padding='same')
        self.conv1b = nn.Conv1d(n_filters[0],n_filters[0],kernel_size,padding='same')
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size)
        
        self.conv2a = nn.Conv1d(n_filters[0],n_filters[1],kernel_size,padding='same')
        self.conv2b = nn.Conv1d(n_filters[1],n_filters[1],kernel_size,padding='same')
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size)
        self.conv3a = nn.Conv1d(n_filters[1],n_filters[2],kernel_size,padding='same')
        self.conv3b = nn.Conv1d(n_filters[2],n_filters[2],kernel_size,padding='same')
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size)
        self.conv4a = nn.Conv1d(n_filters[2],n_filters[3],kernel_size,padding='same')
        self.conv4b = nn.Conv1d(n_filters[3],n_filters[3],kernel_size,padding='same')
        self.act4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size)

        self.conv5a = nn.Conv1d(n_filters[2],n_filters[4],kernel_size,padding='same')
        self.conv5b = nn.Conv1d(n_filters[4],n_filters[4],kernel_size,padding='same')
        self.act5 = nn.ReLU()
        # self.pool5 = nn.AvgPool1d(kernel_size) #is it global?

        ### FC part 
        self.dense1 = nn.Linear(2,n_fc_neurons)
        self.drop1 = nn.Dropout(p=dropout)
        self.dense2 = nn.Linear(n_fc_neurons, n_fc_neurons)
        self.drop2 = nn.Dropout(dropout)
        self.denseOut = nn.Linear(n_fc_neurons, 1)

    
    def forward(self, x):
        x = self.act1(self.conv1a(x))
        print("conv1a")
        x = self.act1(self.conv1b(x))
        x = self.pool1(x)
        print("pool1 finished")
        x = self.act2(self.conv2a(x))
        x = self.act2(self.conv2b(x))
        x = self.pool2(x)
        print("pool2 finished")
        x = self.act3(self.conv3a(x))
        x = self.act3(self.conv3b(x))
        x = self.pool3(x)
        # x = self.act4(self.conv4a(x))
        # x = self.act4(self.conv4b(x))
        # x = self.pool4(x)
        # print("pool4 finished")
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


batchsize = 3
batch_num = 5

### random data
size = batch_num*batchsize
bsp = np.random.rand(size,2,180)
bsp = torch.Tensor(bsp)

bsp_out = np.random.randint(0,2, size=(size,1))
bsp_out = torch.Tensor(bsp_out)


### prepare data
dataset = torch.utils.data.TensorDataset(bsp, bsp_out)
trainloader = torch.utils.data.DataLoader(dataset, batch_size = batchsize, shuffle = False)

### train model
cnn_model = cnn_class()

def train_model(train_dataloader, model, epoch_num):
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(epoch_num):
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            pred = cnn_model(inputs)
            loss = loss_fn(pred, targets)
            loss.backward()
            optimizer.step()


# train_model(trainloader, cnn_model, 3)