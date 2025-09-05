import torch.nn as nn

class WordLSTM(nn.Module):
    def __init__(self, vocab_size, emb=128, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, h=None):
        e = self.emb(x)
        y, h = self.lstm(e, h)
        logits = self.head(y)

        return logits, h
