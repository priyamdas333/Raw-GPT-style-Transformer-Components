import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MiniTransformer(nn.Module):
    """
    The complete tiny transformer.

    Flow: Token Embed + Pos Embed
          → Block 0 (4-head attention + FFN)
          → Block 1 (4-head attention + FFN)
          → LayerNorm → Linear → Logits

    """
    def forward(self,vocab_size=26, d_model=64, n_heads=4,d_ff=128, n_layers=2, max_seq=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq = max_seq

        self.token_emb=nn.Embedding(vocab_size,d_model)
        self.pos_emb=nn.Embedding(max_seq,d_model)

        