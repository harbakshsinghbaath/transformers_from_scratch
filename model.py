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
    
#multi head attention  uwu 
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0 , "d_model is not divisible by h"
        self.d_k = d_model//h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key,value,mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = torch.matmul(query,key.transpose(-2,-1))/ math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores


    def forward(self, q,k,v,mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h,self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttention.attention(query,key,value,mask, self.dropout)

        x= x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.d_k)
        return self.w_o(x)
    


# residual connection 
class ResidualConnection(nn.Module):
    def __init__(self,dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))



class EncoderBlock(nn.Module):

    def __init__(self, self_attention: MultiHeadAttention, feed_forward: FeedForwardLayer, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connection1 = ResidualConnection(dropout)
        self.residual_connection2 = ResidualConnection(dropout)

    def forward(self,x, src_mask):
        x = self.residual_connection1(x,lambda x : self.self_attention(x,x,x,src_mask))
        x = self.residual_connection2(x, self.feed_forward)
        return x
    
#since an encoder can have n encoder blocks
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm  = LayerNormalization()

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
             
    






