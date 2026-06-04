
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmilesCNNEncoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        out_dim: int = 256,
        channels: int = 128,
        kernels=(3, 5, 7),
        dropout: float = 0.2,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=emb_dim, out_channels=channels, kernel_size=k, padding=k // 2)
            for k in kernels
        ])

        self.proj = nn.Sequential(
            nn.Linear(channels * len(kernels), out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim)
        )

    def forward(self, ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.emb(ids)          # [B, L, E]
        x = x.transpose(1, 2)      # [B, E, L]

        feats = []
        for conv in self.convs:
            h = F.relu(conv(x))    # [B, C, L]
            if mask is not None:

                m = mask.unsqueeze(1).to(dtype=h.dtype)
                h = h.masked_fill(m == 0, float("-inf"))
            feats.append(torch.amax(h, dim=-1))

        z = torch.cat(feats, dim=1)
        return self.proj(z)


class CharSmilesTokenizer:
    def __init__(self, smiles_list, pad_token="<pad>", unk_token="<unk>"):
        chars = set()
        for s in smiles_list:
            if s is None:
                continue
            chars.update(list(s))
        self.pad_token = pad_token
        self.unk_token = unk_token
        vocab = [pad_token, unk_token] + sorted(chars)
        self.stoi = {c: i for i, c in enumerate(vocab)}
        self.pad_id = self.stoi[pad_token]
        self.unk_id = self.stoi[unk_token]
        self.vocab_size = len(vocab)

    def encode(self, s, max_len=256):
        if s is None:
            s = ""
        ids = [self.stoi.get(ch, self.unk_id) for ch in s[:max_len]]
        mask = [1] * len(ids)
        if len(ids) < max_len:
            pad_n = max_len - len(ids)
            ids += [self.pad_id] * pad_n
            mask += [0] * pad_n
        return ids, mask

def get_smiles_list(dataset):

    if hasattr(dataset, "smiles"):
        return list(dataset.smiles)

    s_list = []
    for i in range(len(dataset)):
        d = dataset[i]
        if hasattr(d, "smiles"):
            s_list.append(d.smiles)
        else:
            s_list.append(None)
    return s_list