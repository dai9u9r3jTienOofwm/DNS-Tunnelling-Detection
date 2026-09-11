import torch
from torch import nn
class DNSTunnelingCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, num_filters=64, kernel_size=5, dropout=0.3):
        super().__init__(); self.embedding=nn.Embedding(vocab_size,embed_dim,padding_idx=0); self.features=nn.Sequential(nn.Conv1d(embed_dim,num_filters,kernel_size,padding=kernel_size//2),nn.ReLU(),nn.BatchNorm1d(num_filters),nn.MaxPool1d(2),nn.Conv1d(num_filters,num_filters*2,kernel_size,padding=kernel_size//2),nn.ReLU(),nn.AdaptiveMaxPool1d(1)); self.classifier=nn.Sequential(nn.Flatten(),nn.Dropout(dropout),nn.Linear(num_filters*2,1))
    def forward(self,x): return self.classifier(self.features(self.embedding(x).transpose(1,2))).squeeze(1)
