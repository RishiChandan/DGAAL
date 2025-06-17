#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from models import G1, G2, D1, D2, TaskModel


# In[2]:


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(testset, batch_size=64, shuffle=False)


# In[4]:


all_indices = np.arange(len(dataset))
np.random.shuffle(all_indices)
labeled_indices = all_indices[:5000]
unlabeled_indices = all_indices[5000:]


# In[5]:


def get_loader(indices):
    return DataLoader(Subset(dataset, indices), batch_size=64, shuffle=True, drop_last=True)


# In[6]:


latent_dim = 32
g1 = G1(latent_dim).to(DEVICE)
g2 = G2(latent_dim).to(DEVICE)
d1 = D1(latent_dim).to(DEVICE)
d2 = D2().to(DEVICE)
T = TaskModel().to(DEVICE)


# In[7]:


opt_g1 = optim.Adam(g1.parameters(), lr=1e-3)
opt_d1 = optim.Adam(d1.parameters(), lr=1e-3)
opt_g = optim.Adam(list(g1.parameters()) + list(g2.parameters()), lr=3e-3)
opt_d2 = optim.Adam(d2.parameters(), lr=3e-3)
opt_T = optim.Adam(T.parameters(), lr=3e-3)


# In[8]:


bce = nn.BCELoss()
mse = nn.MSELoss()
ce = nn.CrossEntropyLoss()


# In[9]:


def train_gan(epoch=1):
    g1.train()
    g2.train()
    d1.train()
    d2.train()

    for _ in range(epoch):
        loader_L = get_loader(labeled_indices)
        loader_U = get_loader(unlabeled_indices)
        
        for (xL, _), (xU, _) in zip(loader_L, loader_U):
            xL, xU = xL.to(DEVICE), xU.to(DEVICE)

            ### --- FIRST PHASE: Train D2 (image-level discriminator) ---
            with torch.no_grad():
                zL_det = g1(xL).detach()
                xg_det = g2(zL_det).detach()
            d2_real = d2(xL)
            d2_fake = d2(xg_det)
            d2_loss = -torch.mean(d2_real) + torch.mean(d2_fake)
            opt_d2.zero_grad()
            d2_loss.backward()
            opt_d2.step()

            ### --- FIRST PHASE: Train Generator (G1 + G2) with adversarial + pMSE loss ---
            zL = g1(xL)  # No detach here
            xg = g2(zL)
            adv_loss = -torch.mean(d2(xg))  # Fool D2
            pmse_loss = mse(xL, xg)         # Retain semantic closeness
            total_g_loss = adv_loss + 0.1 * pmse_loss
            opt_g.zero_grad()
            total_g_loss.backward()
            opt_g.step()

            ### --- SECOND PHASE: Train D1 (feature-level discriminator) ---
            zL = g1(xL).detach()
            zU = g1(xU).detach()
            d1_real = d1(zL)
            d1_fake = d1(zU)
            d1_loss = -torch.mean(torch.log(d1_real + 1e-8) + torch.log(1 - d1_fake + 1e-8))
            opt_d1.zero_grad()
            d1_loss.backward()
            opt_d1.step()

            ### --- SECOND PHASE: Train G1 (encoder to fool D1) ---
            zU = g1(xU)
            d1_fake = d1(zU)
            g1_loss = -torch.mean(torch.log(d1_fake + 1e-8))
            opt_g1.zero_grad()
            g1_loss.backward()
            opt_g1.step()


# In[10]:


# IMPROVEMENT 1: Adaptive Sampling Strategy
def sample_from_pool(N=500, epoch_fraction=0):
    pool_loader = get_loader(unlabeled_indices)
    scores = []
    images = []
    flat_indices = []
    
    # Calculate diversity scores
    feature_representations = []

    with torch.no_grad():
        for batch_i, (x, _) in enumerate(pool_loader):
            batch_start = batch_i * 64
            batch_indices = unlabeled_indices[batch_start: batch_start + x.size(0)]

            # Get uncertainty scores from D1
            z = g1(x.to(DEVICE))
            p = d1(z).cpu().squeeze()
            
            # Store feature representations for diversity calculation
            feature_representations.append(z.cpu().numpy())
            
            scores.extend(p.tolist())
            images.extend(x)
            flat_indices.extend(batch_indices)

    scores = np.array(scores)
    flat_indices = np.array(flat_indices)
    boundary_weight = min(0.1 + epoch_fraction * 1.8, 1.0)  # Gradually increase from 0.1 to 1.0
    
    # Select top N samples based on weighted score
    # Lower score means higher information content
    final_scores = scores.copy()
    
    # For early epochs, add diversity component
    if epoch_fraction < 0.8:  # Only use diversity in early stages
        all_features = np.vstack(feature_representations)
        mean_feature = np.mean(all_features, axis=0)
        diversity_scores = []
        
        # Calculate distance from mean (higher = more diverse)
        for i, feat in enumerate(all_features):
            dist = np.linalg.norm(feat - mean_feature)
            diversity_scores.append(dist)
        
        diversity_scores = np.array(diversity_scores)
        # Normalize to [0,1] range where lower is better (like uncertainty)
        diversity_scores = 1 - (diversity_scores / np.max(diversity_scores))
        
        # Weighted combination 
        final_scores = boundary_weight * scores + (1 - boundary_weight) * diversity_scores
    
    top_k_indices = np.argsort(final_scores)[:N]
    selected_indices = flat_indices[top_k_indices]
    selected_images = torch.stack([images[i] for i in top_k_indices])

    return selected_indices, selected_images


# In[11]:


def train_T():
    T.train()
    loader = get_loader(labeled_indices)
    for _ in range(3):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = T(x)
            loss = ce(pred, y)
            opt_T.zero_grad(); loss.backward(); opt_T.step()


# In[12]:


def evaluate():
    T.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = T(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")
    return acc


# In[13]:


acc_list = []
label_counts = []

num_rounds = 8
for round in range(num_rounds):
    print(f"\n=== Active Learning Round {round+1} ===")
    train_gan(epoch=1)
    
    # Calculate current training progress for adaptive sampling
    epoch_fraction = round / num_rounds
    
    # Get samples using adaptive sampling
    selected_indices, selected_images = sample_from_pool(500, epoch_fraction)
    
    # Update the labeled and unlabeled sets
    labeled_indices = np.concatenate([labeled_indices, selected_indices])
    unlabeled_indices = np.setdiff1d(unlabeled_indices, selected_indices)
    
    train_T()
    acc = evaluate()
    acc_list.append(acc)
    label_counts.append(len(labeled_indices))


# In[14]:


import matplotlib.pyplot as plt

plt.plot(label_counts, acc_list, marker='o')
plt.xlabel("Number of Labeled Samples")
plt.ylabel("Test Accuracy")
plt.title("DGAAL Accuracy vs Labeled Data")
plt.grid(True)
plt.show()


# In[15]:


new_samples, xU_batch = sample_from_pool(500)

# Save visual samples
import os, torchvision.utils as vutils

round_folder = f"round_{round+1}"
os.makedirs(round_folder, exist_ok=True)

# Save real
vutils.save_image(xU_batch[:25], f"{round_folder}/new_sampled_real_round{round+1}.png", nrow=5, normalize=True)

# Save generated
z = g1(xU_batch[:25].to(DEVICE))
xg = g2(z)
vutils.save_image(xg, f"{round_folder}/new_sampled_generated_round{round+1}.png", nrow=5, normalize=True)

print(f"[✓] Saved sampled_real_round{round+1}.png and sampled_generated_round{round+1}.png")


# In[ ]:




