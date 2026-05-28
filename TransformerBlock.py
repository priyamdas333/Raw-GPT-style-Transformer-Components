import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerBlock(nn.Module):
    #The attention layer mixes information between tokens.
    #The feed-forward layer processes each token independently, giving the model extra nonlinear computation capacity.
    #d_ff controls how powerful/wide that internal MLP is.

    def __init__(self,d_model,n_heads,d_ff):
        super().__init__()
        self.ln1=nn.LayerNorm(d_model)
        self.attention=nn.MHA(d_model,n_heads)
        self.ln2=nn.LayerNorm(d_model)
        self.ff=nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.GELU(),
            nn.Linear(d_ff,d_model),
        )

    def forward(self,x,return_attention=False):
        if return_attention:
            attn_out,attn_weights=self.attention(
                self.ln1(x),return_attention=True
            )
            x=x+attn_out
            x=x+self.ff(self.ln2(x))
            return x,attn_weights
        else:
            x=x+(self.attention(self.ln1(x)))
            x=x+self.ff(self.ln2(self.ln2(x)))
            return x