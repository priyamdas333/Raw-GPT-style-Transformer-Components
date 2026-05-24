import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    
    #Define the parameters of the class for object initialization
    def __init__(self,d_model,d_head):
        super.__init__()
        self.W_Q=nn.Linear(d_model,d_head,bias=False)
        self.W_K=nn.Linear(d_model,d_head,bias=False)
        self.W_V=nn.Linear(d_model,d_head,bias=False)
        self.W_O=nn.Linear(d_head,d_model,bias=False)

    #Define the Methods/Functions over the initialized object
    def forward(self,x):
        
        Q=self.W_Q(x)
        K=self.W_K(x)
        V=self.W_V(x)
        #All the attention computationsare happening in the attention head
        attention_scores=Q @ K.transpose(-2,-1)/math.sqrt(d_head)
        attention_weights=F.softmax(attention_scores)
        attention_output=attention_weights @ V
        #Final output
        O=self.W_O(attention_output)
        return O



