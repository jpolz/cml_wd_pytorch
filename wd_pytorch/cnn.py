import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class ConvBlock1D(nn.Module):
    '''
    tbd
    '''

    def __init__(self, dim, kernel_size):
        super().__init__()
        self.conv1d1 = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2,)
        # self.norm = LayerNorm(dim, eps=1e-6)
        self.act = nn.GELU()
        self.conv1d2 = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2,)

    def forward(self, x):
        input = x
        x = self.conv1d1(x)
        # x = self.norm(x)
        x = self.act(x)
        x = input + x

        return x
    
class MFBlock1D(nn.Module,):
    '''
    tbd
    '''

    def __init__(self, dim, ksizes):
        super().__init__()
        
        self.block1 = nn.ModuleList(
            [ConvBlock1D(dim, kernel_size) for kernel_size in ksizes],
        )

    def forward(self, x):
        x = [c(x) for c in self.block1]
        x = torch.cat(x, dim=1)
        return x

class WDConv(nn.Module):
    '''
    tbd
    '''

    def __init__(self, dim, ksizes):
        super().__init__()
        self.ksizes = ksizes
        self.nblocks = 3
        self.blocks = nn.ModuleList(
            [MFBlock1D(dim*(len(ksizes)**i), ksizes) for i in range(self.nblocks)],
        )
        # self.pool1 = nn.MaxPooling1D()
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


if __name__== '__main__':
    model = WDConv(dim = 2, ksizes=[1,3,5,7])
    src = torch.rand((10,2,180,))  # (batch_size, seq_len)
    output = model(src)
    print(output.shape)
