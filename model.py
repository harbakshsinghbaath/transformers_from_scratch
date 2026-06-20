import torch 
import torch.nn as nn
import math

# the first step is creating input embeddings of our vocabulary
class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int ,vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size= vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
# now we will add our positional encoding to our input embeddings
class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        #matrix ofshape (seq_len, d_model) to hold the positional encodings
        pe = torch.zeros(seq_len, d_model)
        # actual postional encoding is calculated using sine and cosine functions of different frequencies
        position = torch.arange(0,seq_len, dtype= torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self,eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x-mean)/(std+self.eps) + self.bias 

class FeedForwardLayer(nn.Module):
    def __init__(self,d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        self.x = self.linear1(x)
        self.x = torch.relu(self.x)
        self.x = self.dropout(self.x)
        self.x = self.linear2(self.x)
        return self.x


