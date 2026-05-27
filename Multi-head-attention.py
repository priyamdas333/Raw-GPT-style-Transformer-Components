import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MHA(nn.Module):

    def __init__(self,d_model,n_heads):
        super.__init__()
        self.d_head=n_heads//d_model
        self.n_heads=n_heads
        self.heads=nn.ModuleList([
            SelfAttention(d_model,self.d_head)
            for _ in range(n_heads)
        ])

    def forward(self,x, return_attention=False):
        if return_attention:
            outputs,attentions=[],[]
            for head in self.heads:
                output,attention= head(x,return_attention=True)
                outputs.append(output)
                attentions.append(attention)
                return sum(outputs),torch.stack(attentions,dim=0)
        else:
            return sum(head(x) for head in self.heads)
