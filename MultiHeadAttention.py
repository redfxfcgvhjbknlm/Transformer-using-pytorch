class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k      = d_model // num_heads
        self.h        = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _attend(self, Q, K, V, mask=None):
        scale  = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)

        def project_split(x, W):
            # (B, seq, d_model) → (B, h, seq, d_k)
            return W(x).view(B, -1, self.h, self.d_k).transpose(1, 2)

        Q = project_split(Q, self.W_q)
        K = project_split(K, self.W_k)
        V = project_split(V, self.W_v)

        out = self._attend(Q, K, V, mask)            # (B, h, seq, d_k)
        out = out.transpose(1, 2).contiguous()        # (B, seq, h, d_k)
        out = out.view(B, -1, self.h * self.d_k)    # (B, seq, d_model)
        return self.W_o(out)
