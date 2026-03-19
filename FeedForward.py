class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),   # expand:  512 → 2048
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),   # contract: 2048 → 512
        )

    def forward(self, x):
        return self.net(x)
