import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 512  # change it to fit your memory constraints, e.g., 256, 128 if you run out of memory
EPOCHS = 5
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, skipgram_df):
        self.centers = torch.tensor(skipgram_df['center'].values, dtype=torch.long)
        self.contexts = torch.tensor(skipgram_df['context'].values, dtype=torch.long)

    def __len__(self):
        return len(self.centers)
    
    def __getitem__(self, index):
        return self.centers[index], self.contexts[index]

# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        # Create embedding layers
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_idx, context_idx):
        # Get embeddings (2D tensors; each row is a word vector)
        center_vecs = self.in_embed(center_idx)             # (batch size, embedding dim)
        context_vecs = self.out_embed(context_idx)          # (batch size, embedding dim)

        # Dot product
        scores = (center_vecs * context_vecs).sum(dim=1)    # (batch size,)
        return scores

# Load processed data
with open('student/Assignment_2/processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Precompute negative sampling distribution below
## Get word frequency counts
counter = data.get('counter', {})

## Create tensor of word counts aligned w/ vocab idx
vocab_size = len(data['word2idx'])
word_counts = torch.zeros(vocab_size, dtype=torch.float) 
## Handle cases of words not in freq dict
for word, idx in data['word2idx'].items():
    ## count = 0 if word not in counter; set to float for 3/4 power
    word_counts[idx] = float(counter.get(word, 0))

## Apply 3/4 power smoothing to word counts + normalize
neg_samp_dist = word_counts.pow(0.75)
neg_samp_dist = neg_samp_dist / neg_samp_dist.sum()

# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():       # Note CUDA is GPU
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Dataset and DataLoader
dataset = SkipGramDataset(data['skipgram_df'])
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, Loss, Optimizer
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def make_targets(center, context, vocab_size):
    '''
    Turns each positive pair to 1+NEGATIVE_SAMPLES training samples

    center: batch of center word indices
    context: batch of true context word indices
    '''
    # Positive examples
    pos_centers = center
    pos_contexts = context
    pos_labels = torch.ones(center.size(0), device=center.device)

    # Negative sampling
    neg_contexts = torch.multinomial(
        neg_samp_dist, 
        num_samples = center.size(0) * NEGATIVE_SAMPLES,
        replacement = True  # Same neg word can appear multiple times
    ).to(center.device)

    neg_centers = center.repeat_interleave(NEGATIVE_SAMPLES)    # Repeat each center idx NEGATIVE_SAMPLES times to get matching centers
    neg_labels = torch.zeros(center.size(0) * NEGATIVE_SAMPLES, device=center.device)

    # Combine centers, contexts, and labels
    all_centers = torch.cat([pos_centers, neg_centers])
    all_contexts = torch.cat([pos_contexts, neg_contexts])
    all_labels = torch.cat([pos_labels, neg_labels])

    return all_centers, all_contexts, all_labels

# Training loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0

    # Iterate over dataset in batches
    step = 0
    for centers, contexts in loader:
        # Move tensors to device
        centers = centers.to(device)
        contexts = contexts.to(device)

        # Expand each positive example
        all_centers, all_contexts, all_labels = make_targets(centers, contexts, vocab_size)

        # Get embeddings + compute dot products (raw scores)
        logits = model(all_centers, all_contexts)

        # Apply BCEWithLogitsLoss (sigmoid + binary cross-entropy) to get combined pos & neg loss
        loss = criterion(logits, all_labels)

        optimizer.zero_grad()       # Clear old gradients
        loss.backward()             # Backpropagate
        optimizer.step()            # Update embeddings w/ Adam

        # Get scalar loss value and add to epoch total
        total_loss += loss.item() 

        # Print progress
        step += 1
        if step % 10000 == 0:
            print(f'Epoch {epoch+1} {100*step/len(loader):.2f}% complete.')
    
    # Print epoch loss
    print(f'Epoch {epoch+1}: Average loss = {total_loss/len(loader)}')

# Save embeddings and mappings
embeddings = model.get_embeddings()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
