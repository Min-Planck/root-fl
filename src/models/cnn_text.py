import torch
import torch.nn as nn
import torch.nn.functional as F

#https://arxiv.org/pdf/1408.5882
class CNN_Text_Header(nn.Module):
    def __init__(self, embed_num=2000, embed_dim=8, kernel_sizes=[3, 4, 5], class_num=4, kernel_num=32, dropout=0.5):
        super(CNN_Text_Header, self).__init__()

        V = embed_num      
        D = embed_dim     
        C = class_num      
        Ci = 1                 
        Co = kernel_num     
        Ks = kernel_sizes   

        self.embed = nn.Embedding(V, D)

        # conv2d input: (N, Ci=1, W, D)
        # output: (N, Co, W-K+1, 1) → squeeze(3) → (N, Co, W-K+1)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.embed(x)           # (N, W, D)
        x = x.unsqueeze(1)          # (N, Ci=1, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W-K+1), ...]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]   # [(N, Co), ...]

        x = torch.cat(x, 1)         # (N, Co * len(Ks))
        x = self.dropout(x)
        return x

class CNN_Text(nn.Module): 
    def __init__(self, embed_num=2000, embed_dim=8, kernel_sizes=[3, 4, 5], class_num=4, kernel_num=32, dropout=0.5): 
        super(CNN_Text, self).__init__()
        self.encode = CNN_Text_Header(embed_num, embed_dim, kernel_sizes, class_num, kernel_num, dropout)
        self.classification = nn.Linear(len(kernel_sizes) * kernel_num, class_num)
    
    def forward(self, x):
        x = self.encode(x)
        x = self.classification(x)
        return x