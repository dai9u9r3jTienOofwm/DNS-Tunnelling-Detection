import re, string, torch
from torch.utils.data import Dataset
VOCAB = ['<pad>', '<unk>'] + list(string.ascii_lowercase + string.digits + '.-_')
CHAR2ID = {c:i for i,c in enumerate(VOCAB)}
def normalize_domain(domain: str) -> str: return domain.strip().lower().rstrip('.')
def encode_domain(domain: str, max_len: int = 253):
    ids=[CHAR2ID.get(c,1) for c in normalize_domain(domain)[:max_len]]
    return ids + [0]*(max_len-len(ids))
class DNSTunnelingDataset(Dataset):
    def __init__(self, domains, labels=None, max_len=253): self.domains=list(domains); self.labels=None if labels is None else list(labels); self.max_len=max_len
    def __len__(self): return len(self.domains)
    def __getitem__(self, idx):
        x=torch.tensor(encode_domain(self.domains[idx],self.max_len),dtype=torch.long)
        return (x, torch.tensor(self.labels[idx],dtype=torch.float32)) if self.labels is not None else x
