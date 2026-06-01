import torch
from torch.utils.data import Dataset, DataLoader
import string
class RepeatedPatternDataset(Dataset):
    def __init__(self, num_samples=5000, pattern_len=3, num_repeats=4,
                 vocab_size=26, seed=42):
        
        self.num_samples=num_samples
        self.pattern_len = pattern_len
        self.num_repeats = num_repeats
        self.seq_len = pattern_len * num_repeats
        self.vocab_size = vocab_size

        self.chars=list(string.ascii_lowercase[:26])
        self.char_to_idx={ch: i for i,ch in enumerate(self.chars)}
        self.idx_char={i: ch for i,ch in enumerate(self.chars)}

        random.seed(seed)
        self.data = []
        for _ in range(num_samples):
            pattern = [random.randint(0, vocab_size - 1)
                       for _ in range(pattern_len)]
            self.data.append(pattern * num_repeats)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.data[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)  # Input
        y = torch.tensor(seq[1:],  dtype=torch.long)  # Target
        return x, y

    def decode(self, indices):
        """Convert list of ints to string."""
        return ''.join(self.idx_to_char[i] for i in indices)
