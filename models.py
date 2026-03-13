import torch
import torch.nn as nn
from constants import VOCAB_SIZE, DEVICE, char2idx, idx2char, PAD, MAX_LEN

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, n_heads: int = 4, n_layers: int = 2, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos = nn.Embedding(512, emb_dim)
        self.dropout = nn.Dropout(0.1)

        enc_layer = nn.TransformerEncoderLayer(emb_dim, n_heads, dropout=0.1, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(emb_dim, n_heads, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)

        self.fc = nn.Linear(emb_dim, vocab_size)
        self.fc.weight = self.emb.weight

    def add_pos(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        return self.emb(x) + self.pos(positions)

    @staticmethod
    def causal_mask(sz, device):
        return torch.triu(torch.ones((sz, sz), dtype=torch.bool, device=device), diagonal=1)

    def forward(self, src, tgt):
        src_emb = self.dropout(self.add_pos(src))
        tgt_emb = self.dropout(self.add_pos(tgt))

        tgt_mask = self.causal_mask(tgt.size(1), tgt.device)
        src_key_padding_mask = (src == self.pad_idx)
        tgt_key_padding_mask = (tgt == self.pad_idx)

        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=src_key_padding_mask)
        return self.fc(out)

class FlowNet(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=char2idx[PAD])
        self.pos = nn.Embedding(MAX_LEN + 1, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dropout=0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(enc_layer, n_layers)
        self.fc = nn.Linear(d_model, VOCAB_SIZE)
        self.fc.weight = self.emb.weight
        self.logZ = nn.Parameter(torch.zeros(()))  # Learned normalizer

    def causal_mask(self, sz):
        return torch.triu(torch.ones((sz, sz), dtype=torch.bool, device=DEVICE), diagonal=1)

    def forward(self, prefix):
        B, L = prefix.shape
        pos = torch.arange(L, device=prefix.device).unsqueeze(0).expand(B, L)
        x = self.emb(prefix) + self.pos(pos)

        # We need to construct mask with float('-inf') or bool for TransformerEncoder
        # PyTorch TransformerEncoder with batch_first=True takes mask of shape (L, L)
        mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=prefix.device), diagonal=1)

        # pass explicitly is_causal=True if supported, else pass mask
        x = self.tr(x, mask=mask)
        logits = self.fc(x[:, -1])  # Logits for next token (B, VOCAB_SIZE)
        return torch.log_softmax(logits, dim=-1)  # Log probs
