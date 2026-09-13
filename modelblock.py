import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import random 
from modelblock import Block 
class Block(nn.Module):
    def __init__(self, embedding_dim, block_size, n_head):
        super().__init__()
import sentencepiece as spm

print ("torch version:", torch.__version__) 
print ("CUDA available", torch.cuda.is_available()) 
print ("gpu name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no")

# Train tokenizer
with open("corpus.txt","r", encoding="utf-8") as f: 
    text = f.read() 
    
spm.SentencePieceTrainer.Train( 
    input="corpus.txt", model_prefix="tokenizer", vocab_size=100, model_type="bpe") 
    
sp = spm.SentencePieceProcessor() 
sp.load("tokenizer.model") 

ids = sp.encode(text, out_type=int) 
data = torch.tensor(ids, dtype=torch.long) 
print(data) 
vocab_size = sp.get_piece_size() 
print(vocab_size)

# Hyperparameters
block_size = 6 
embedding_dim = 32 
n_head = 2 
n_layers = 2 
lr = 1e-3 
epochs = 1500 

def get_batch(batch_size=16): 
    ix = torch.randint(0, len(data) - block_size, (batch_size,)) 
    x = torch.stack([data[i:i+block_size] for i in ix]) 
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) 
    return x, y

class tinygpt(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim) 
        self.position_embedding = nn.Embedding(block_size, embedding_dim) 
        # Note: If your custom Block class uses a config object, wrap these in a configuration class
        self.block = nn.Sequential(*[Block(embedding_dim, block_size, n_head) for _ in range(n_layers)]) 
        self.ln_f = nn.LayerNorm(embedding_dim) 
        self.head = nn.Linear(embedding_dim, vocab_size) 
        
    def forward(self, idx, targets=None): 
        B, T = idx.shape 
        tok_emb = self.token_embedding(idx) 
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device)) 
        x = tok_emb + pos_emb 
        x = self.block(x) 
        x = self.ln_f(x) 
        logits = self.head(x) 
        
        loss = None 
        B, T, C = logits.shape 
        if targets is not None: 
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T)) 
        return logits, loss 
        
    def generate(self, idx, max_new_tokens): 
        for _ in range(max_new_tokens): 
            idx_cond = idx[:, -block_size:] 
            logits, _ = self(idx_cond) 
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

# Training loop
model = tinygpt() 
optimizer = torch.optim.AdamW(model.parameters(), lr=lr) 

for epoch in range(epochs): 
    xb, yb = get_batch() 
    logits, loss = model(xb, yb) 
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 
    if epoch % 300 == 0: 
        print(f"Step {epoch}, loss={loss.item():.4f}")

# Inference
context = torch.tensor([sp.encode("help")], dtype=torch.long) 
out = model.generate(context, max_new_tokens=20) 
print("\nGenerated text:\n") 
generated_ids = out[0].tolist() 
print(sp.decode(generated_ids))
