"""CharLM benchmark: matches examples/charlm/main.zig
1-layer Transformer (pre-norm), embed_dim=64, seq_len=32, vocab=28, ff=256
"""
import time
import torch
import torch.nn as nn
import math

VOCAB_SIZE = 28
EMBED_DIM = 64
SEQ_LEN = 32
FF_DIM = 4 * EMBED_DIM

def char_encode(c):
    if c == ' ': return 0
    if c == '.': return 1
    if 'a' <= c <= 'z': return ord(c) - ord('a') + 2
    return 0

def char_decode(idx):
    if idx == 0: return ' '
    if idx == 1: return '.'
    if 2 <= idx <= 27: return chr(idx - 2 + ord('a'))
    return '?'

class CharLMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_emb = nn.Parameter(torch.randn(SEQ_LEN, EMBED_DIM) * (1.0 / math.sqrt(SEQ_LEN)))
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.attn = nn.MultiheadAttention(EMBED_DIM, 1, batch_first=True)
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.ff1 = nn.Linear(EMBED_DIM, FF_DIM)
        self.ff2 = nn.Linear(FF_DIM, EMBED_DIM)
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.out_proj = nn.Linear(EMBED_DIM, VOCAB_SIZE)
        self.register_buffer('causal_mask',
            torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool())

    def forward(self, x):
        h = self.tok_emb(x) + self.pos_emb
        h2 = self.ln1(h)
        sa, _ = self.attn(h2, h2, h2, attn_mask=self.causal_mask)
        h = h + sa
        h2 = self.ln2(h)
        ff = torch.nn.functional.gelu(self.ff1(h2))
        h = h + self.ff2(ff)
        h = self.ln_f(h)
        return self.out_proj(h)

def prepare_corpus():
    base_text = "hello world. "
    corpus = base_text * 40
    num_seq = min((len(corpus) - SEQ_LEN - 1) // (SEQ_LEN // 2), 128)
    input_ids = torch.zeros(num_seq, SEQ_LEN, dtype=torch.long)
    target_ids = torch.zeros(num_seq, SEQ_LEN, dtype=torch.long)
    for s in range(num_seq):
        for t in range(SEQ_LEN):
            pos = s * (SEQ_LEN // 2) + t
            input_ids[s, t] = char_encode(corpus[pos])
            target_ids[s, t] = char_encode(corpus[pos + 1])
    return input_ids, target_ids, num_seq, corpus[:len(base_text)], len(corpus)

def run(device):
    torch.manual_seed(42)
    input_ids, target_ids, num_seq, sample, corpus_len = prepare_corpus()
    print(f'  Corpus: "{sample}..." ({corpus_len} chars)')
    print(f'  Training sequences: {num_seq} (seq_len={SEQ_LEN}, overlap=50%)')

    model = CharLMModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Model: 1-layer Transformer, {total_params} params (~{total_params*4//1024}KB)\n')

    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 200

    start = time.perf_counter()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for s in range(num_seq):
            optimizer.zero_grad()
            logits = model(input_ids[s:s+1])  # [1, SEQ_LEN, VOCAB]
            loss = nn.functional.cross_entropy(
                logits.view(-1, VOCAB_SIZE), target_ids[s].view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            avg = epoch_loss / num_seq
            print(f'  Epoch {epoch:>4}: loss = {avg:.4f}')

    elapsed = time.perf_counter() - start
    print(f'\n  Training time: {elapsed*1000:.0f}ms')

    # Generate text
    model.eval()
    seed = "hello world. hello world. hello"
    gen_ctx = [0] * SEQ_LEN
    for i, c in enumerate(seed):
        gen_ctx[SEQ_LEN - len(seed) + i] = char_encode(c)
    result = list(seed)
    with torch.no_grad():
        for _ in range(60):
            inp = torch.tensor([gen_ctx], dtype=torch.long, device=device)
            logits = model(inp)
            pred = logits[0, -1].argmax().item()
            result.append(char_decode(pred))
            gen_ctx = gen_ctx[1:] + [pred]
    print(f'\n  Generated text:\n    "{"".join(result)}"')

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    device = "cuda" if mode == "cuda" else "cpu"
    print(f"=== CharLM (PyTorch {mode.upper()}: Transformer + Adam) ===\n")
    run(device)
