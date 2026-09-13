import torch
import torch.nn as nn
import torch.nn.functional as F
import random 

import sys
sys.stdout.reconfigure(encoding='utf-8')

class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size, n_embd, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5) # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Perform weighted aggregation of values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v    # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ A simple linear layer followed by non-linearity """
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication (attention) followed by computation (feedforward) """
    def __init__(self, n_embd, n_head, block_size=256, dropout=0.0):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

print ("torch version:", torch.__version__)
print ("CUDA available", torch.cuda.is_available())
print ("gpu name", torch.cuda.get_device_name(0)if torch.cuda.is_available() else "no")

import sentencepiece as spm

with open("corpus.txt","r", encoding="utf-8") as f:
    text = f.read()

spm.SentencePieceTrainer.Train(
    input="corpus.txt",
    model_prefix="tokenizer",
    vocab_size=1000,
    model_type="bpe")

sp= spm.SentencePieceProcessor()
sp.load("tokenizer.model")

ids = sp.encode(text, out_type=int)
data = torch.tensor(ids,dtype=torch.long)
print(data)

vocab_size = sp.get_piece_size()
print(vocab_size)

#corpus =[
   # I	made	some	changes	when	I	wrote	the	stories	in	this	book	but	mostly	it	is	a true	reflection	of	my	childhood. When	my	granddaughter	Krishnaa	was	born,	she	elevated	me	to	the position	of	grandmother."
    #he	orphan	boy	outwits	his	greedy	uncles with	a	bag	of	ash;	and	an	old	couple	in	distress	is	saved	by	a	magic	drum. Sudha	Murty’s	grandparents	told	her	some	of	these	stories	when	she	was	a child;	others	she	heard	from	her	friends	from	around	the	world.	These delightful	and	timeless	folk	tales	have	been	her	favourites	for	years,	and	she has	recounted	them	many	times	over	to	the	young	people	in	her	life.	With	this collection,	they	will	be	enjoyed	by	many	more	readers,	of	all	ages."
    #
#corpus =[s+" <end>" for s in corpus]
#text = " ". join (corpus)


#n(data))

block_size = 2
embedding_dim = 32
n_head=2
n_layers=2
lr= 1e-3
epoch = 1500

def get_batch(batch_size=16):
    ix=torch.randint(0,len(data)-block_size,(batch_size,))
    x= torch.stack([data[i:i+block_size] for i in ix])
    y= torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x , y 

class tinygpt(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding (vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding (block_size, embedding_dim)
        self.block= nn.Sequential(*[Block(embedding_dim,block_size, n_head) for _ in range (n_layers)])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim,vocab_size)

    def forward(self, idx, targets=None):
        B , T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T,device=idx.device))
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
        # If context is longer than block size, keep only the last block_size tokens
           idx_cond = idx[:, -block_size:]

        # Get logits for this context
           logits, _ = self(idx_cond)          # forward()

        # Focus on logits for the last time step
           logits = logits[:, -1, :]      # shape: (B, vocab_size)

        # Convert logits to probabilities
           probs = F.softmax(logits, dim=-1)

        # Sample next token ID from the distribution
           idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

        # Append sampled token to the running sequence
           idx = torch.cat((idx, idx_next), dim=1)             # (B, T+1)

        return idx

model = tinygpt()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for epoch in range(epoch):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 300 == 0:
        print(f"Step{epoch}, loss={loss.item():.4f}")

# training loop finish above
#context= torch.tensor([[word2idx["help"]]],dtype=torch.long)
#out = model.generate(context, max_new_tokens=15)

#print(" ". join([idx2word[int(i)] for i in out[0]]))

# Print raw output tokens to verify it generated 15 new tokens
# print("Raw generated IDs:", out[0].tolist())

# # Convert IDs to words (and handle unknown tokens safely)
# words = [idx2word.get(int(i), "<UNK>") for i in out[0]]
# print("Generated text:", " ".join(words))

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

context = torch.tensor([sp.encode("help")],dtype=torch.long)

out= model.generate(context,max_new_tokens=20)

print("Generated text:\n")
generated_ids = out[0].tolist()
print(sp.decode(generated_ids).encode('utf-8',errors='ignore').decode('utf-8'))