# -----------------------------------------------------------------------------
# Copyright (c) 2025 Mahesh Godavarti. All Rights Reserved.
#
# License: This software is provided for non-commercial research purposes only.
# Any commercial use, including but not limited to use in a product, service,
# or for-profit research, is strictly prohibited without explicit written
# permission from the copyright holder.
#
# Patent Pending: Certain aspects of this software are the subject of a
# pending patent application.
#
# Contact: m@qalaxia.com
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import  functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
#hyperparams
epochs = 100
lr = 5e-4
batch_size = 32
block_size = 20
device = "cuda" if torch.cuda.is_available() else "cpu"
max_iters = 20000
print_iters = 100
eval_iters = 10
eval_interval = 300
n_embed=40
n_layers = 1
dropout = 0.2
# ---------
with open("input.txt", "r") as f:
  text = f.read()

# Unique characters
chars = sorted(list(set(text)))
print(''.join(chars))
vocab_size = len(chars)
print(vocab_size)

#Tokenizers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda xx: [stoi[x] for x in xx]
decode = lambda xx: ''.join([itos[x] for x in xx])
encode("Hello!")
print(decode(encode("Hello!")))


# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  # generate targets and context
  if split == "train":
    data = train_data
  else:
    data = val_data
  index = torch.randint(0,len(data)-block_size,(batch_size,))
  x = torch.stack([data[ind:ind+block_size] for ind in index])
  y = torch.stack([data[ind+1:ind+block_size+1] for ind in index])
  return x.to(device),y.to(device)


@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'test']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X,Y = get_batch(split)
      logits, loss = model(X,Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

class LayerNorm(nn.Module):
  def __init__(self, dim) -> None:
    super().__init__()
    self.eps = 1e-5
    # params
    self.gamma = nn.Parameter(torch.ones(dim))
    self.beta = nn.Parameter(torch.zeros(dim))

  def forward(self, x):
    xmean = x.mean(dim=1, keepdim=True)
    xvar = ((x-xmean)**2).mean(dim=1, keepdim=True)
    xhat = (x-xmean) / torch.sqrt(xvar + self.eps)
    self.out = self.gamma * xhat + self.beta
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]

class Attention(nn.Module):
  def __init__(self, n_embed) -> None:
    super().__init__()
    self.keys = nn.Linear(n_embed, n_embed)
    self.queries = nn.Linear(n_embed, n_embed)
    self.values = nn.Linear(n_embed, n_embed)
    self.n_embed = n_embed
    self.register_buffer('tril', torch.tril(torch.ones((block_size,block_size))).to(device))
    self.dropout = nn.Dropout(dropout)
    self.proj = nn.Linear(n_embed, n_embed)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape
    k = self.keys(x) # (B,T,C_h)
    q = self.queries(x) # (B,T,C_h)
    v = self.values(x) # (B,T,C_h)

    angle1 = torch.arange(T, device=device)
    angle2 = torch.arange(int(C/2), device=device)
    angle = torch.outer(angle1, angle2)
    angle = angle.unsqueeze(0)
    angle = torch.flip(angle, dims=(1,))
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    cos_angle = cos_angle.unsqueeze(3)
    sin_angle = sin_angle.unsqueeze(3)
    top_row = torch.cat((cos_angle, sin_angle), dim=3)
    bottom_row = torch.cat((-sin_angle, cos_angle), dim=3)
    top_row = top_row.unsqueeze(4)
    bottom_row = bottom_row.unsqueeze(4)
    matrix = torch.cat((top_row, bottom_row), dim=4)

    kmod = k.reshape(B, T, int(C/2), 2, 1)
    kmod = torch.matmul(matrix, kmod)
    k = kmod.reshape(B, T, C)

    qmod = q.reshape(B, T, int(C/2), 2, 1)
    qmod = torch.matmul(matrix, qmod)
    q = qmod.reshape(B, T, C)

    #vmod = v.reshape(B, T, int(C/2), 2, 1)
    #vmod = torch.matmul(matrix, vmod)
    #v = vmod.reshape(B, T, C)

    wei = k @ q.transpose(-1,-2) * C**(-0.5)# (B,T,T)
    wei = wei.masked_fill( self.tril[:T,:T]==0, float('-inf'))
    # wei = F.softmax(wei, dim=-1) # (B,T,T)
    wei = torch.log(torch.exp(wei)+1) # (B,T,T)
    wei = self.dropout(wei)
    out = wei @ v # (B,T,C_h)

    #outmod = out.reshape(B, T, int(C/2), 2, 1)
    #outmod = torch.matmul(torch.transpose(matrix, 3, 4), outmod)
    #out = outmod.reshape(B, T, C)

    out = self.proj(out)
    out = self.dropout(out)
    return out

class FeedForward(nn.Module):
  def __init__(self, n_embed) -> None:
    super().__init__()
    self.ffn = nn.Sequential(
      nn.Linear(n_embed, 4*n_embed),
      nn.ReLU(),
      nn.Linear(4*n_embed, n_embed),
      nn.Dropout(dropout),
    )
  def forward(self, x):
    return self.ffn(x)

class Block(nn.Module):
  def __init__(self, n_embed) -> None:
    super().__init__()
    self.sa_head = Attention(n_embed)
    self.ffn = FeedForward(n_embed)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)


  def forward(self, x):
    x = x + self.sa_head(self.ln1(x))
    x = x + self.ffn(self.ln2(x))

    return x

class BigramNeuralNetwork(nn.Module):
  def __init__(self,vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size,n_embed)
    self.position_embedding_table = nn.Embedding(block_size,n_embed)
    self.lm_head = nn.Linear(n_embed,vocab_size)
    self.ffn = FeedForward(n_embed)
    self.blocks = nn.Sequential(*[Block(n_embed) for _ in range(n_layers)])


  def forward(self, idx, targets=None):
    # idx = idx[:,-block_size:]
    B,T = idx.shape
    x = self.token_embedding_table(idx) # (B,T,C_e)
    x = self.blocks(x) # (B,T,C_e)
    logits = self.lm_head(x) # (B,T,vocab_size)
    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T,C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
      logits = logits.view(B,T,C)
    return logits, loss
  def generate(self, idx, max_new_tokens):
    # idx is (B,T)
    idx_next = []
    for i in range(max_new_tokens):
      idx_cond = idx[:,-block_size:]
      logits, loss = self(idx_cond)
      last_timestep = logits[:,-1,:]
      probs = F.softmax(last_timestep, dim=1)
      next_index = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, next_index), dim=1)
    for arr in idx:
      print(decode(arr.cpu().detach().numpy()))
    return idx

model = BigramNeuralNetwork(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(),lr=lr)

# checkpoint = torch.load('model.pt')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
checkpoint_path = None#"./differentattention/model_40.pt"
epoch = 0
if checkpoint_path:
  checkpoint = torch.load(checkpoint_path)
  print(checkpoint)
  if checkpoint['model_state_dict']:
    model.load_state_dict(checkpoint['model_state_dict'].to(device))
  if checkpoint['optimizer_state_dict']:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
device = "cuda" if torch.cuda.is_available() else "cpu"
m = model.to(device)
print("Uses device " + device)
MODEL_CHECKPOINT = "./differentattention/model_{iter}.pt"
losses_data = {"train":[], "test":[]}
for iter in tqdm(range(epoch ,max_iters)):
  if iter % eval_iters == 0:
    losses = estimate_loss()
    losses_data['train'].append(losses['train'].cpu().numpy())
    losses_data['test'].append(losses['test'].cpu().numpy())
    print(f"Step {iter}, train loss:{losses['train']:.4f}, test loss:{losses['test']:.4f}")

  if iter % print_iters == 0:
    losses = estimate_loss()
    torch.save({
            'epoch': iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses,
            }, MODEL_CHECKPOINT.format(iter=iter))
    losses_data['train'].append(losses['train'].cpu().numpy())
    losses_data['test'].append(losses['test'].cpu().numpy())
    model.eval()
    with torch.no_grad():
      #Generate from the model:
      output = m.generate(torch.zeros((1,2), dtype=torch.long).to(device).contiguous(), 1000  )[0].tolist()

    print(f"Step {iter}, train loss:{losses['train']:.4f}, test loss:{losses['test']:.4f}")
    model.train()

  #Get data
  xb,yb = get_batch("train")

  #Evaluate loss
  logits,loss = model(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.0)
  optimizer.step()
torch.save(model.state_dict(), "./differentattention/model.pt")
#Generate from the model:
output = m.generate(torch.zeros((1,2), dtype=torch.long).to(device),1000)[0].tolist()

