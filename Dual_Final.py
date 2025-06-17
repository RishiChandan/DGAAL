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
import os, torchvision.utils as vutils
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


# IMPROVEMENT 2: Progressive Image Generation
def train_gan(epoch=1, current_epoch=0, total_epochs=5):
    g1.train()
    g2.train()
    d1.train()
    d2.train()
    
    # Progressive noise schedule - increase noise as training progresses
    noise_factor = min(0.05 + (current_epoch / total_epochs) * 0.15, 0.2)  # Scale from 0.05 to 0.2

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
            
            # Add progressive noise to the latent representation
            if noise_factor > 0:
                noise = torch.randn_like(zL) * noise_factor
                zL_noisy = zL + noise
            else:
                zL_noisy = zL
                
            xg = g2(zL_noisy)
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
    labels = []  # Added to store the predicted labels for class balancing
    
    # Calculate diversity scores
    feature_representations = []

    with torch.no_grad():
        for batch_i, (x, y) in enumerate(pool_loader):
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
            labels.extend(y.numpy())  # Store labels for class balancing

    scores = np.array(scores)
    flat_indices = np.array(flat_indices)
    labels = np.array(labels)
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
    
    # Also return the selected labels for class balancing
    selected_labels = [labels[i] for i in top_k_indices]
    
    return selected_indices, selected_images, selected_labels


# In[11]:


# IMPROVEMENT 3: Class-Balanced Generation
def generate_balanced_samples(selected_images, selected_labels, noise_factor=0.1):
    # Count class distribution in the labeled pool
    class_counts = {}
    for idx in labeled_indices:
        label = dataset[idx][1]
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    
    # Count classes in currently selected samples
    for label in selected_labels:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    
    # Identify underrepresented classes
    total_samples = sum(class_counts.values())
    class_frequencies = {cls: count/total_samples for cls, count in class_counts.items()}
    
    # Calculate inverse frequency weights (higher weight for rare classes)
    if len(class_frequencies) > 0:  # Ensure we have class data
        inv_frequencies = {cls: 1.0/max(freq, 0.01) for cls, freq in class_frequencies.items()}
        max_inv_freq = max(inv_frequencies.values())
        normalized_weights = {cls: inv_freq/max_inv_freq for cls, inv_freq in inv_frequencies.items()}
    else:
        normalized_weights = {}
    
    # Group samples by class
    class_samples = {}
    for i, label in enumerate(selected_labels):
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append(selected_images[i])
    
    generated_images = []
    generated_labels = []
    
    # Generate more samples for underrepresented classes
    for label, samples in class_samples.items():
        # Determine number of samples to generate based on class weight
        weight = normalized_weights.get(label, 1.0)  # Default to 1.0 if class not found
        # More for underrepresented classes, base is 5 samples per class
        num_to_generate = max(5, int(20 * (1.0 - class_frequencies.get(label, 0.1))))
        
        # If we have samples for this class, generate new ones
        if samples:
            for _ in range(num_to_generate):
                # Randomly select a sample from this class
                if len(samples) > 0:
                    idx = np.random.randint(0, len(samples))
                    x = samples[idx].to(DEVICE)
                    
                    # Generate new sample with appropriate noise level
                    with torch.no_grad():
                        z = g1(x.unsqueeze(0))
                        # More noise for rare classes
                        current_noise = noise_factor * (1.0 + weight)
                        z_noisy = z + torch.randn_like(z) * current_noise
                        x_gen = g2(z_noisy)
                    
                    generated_images.append(x_gen.squeeze(0).cpu())
                    generated_labels.append(label)
    
    print(f"Generated {len(generated_images)} additional samples for class balancing")
    
    if len(generated_images) > 0:
        return torch.stack(generated_images), generated_labels
    else:
        return None, []


# In[12]:


def train_T():
    T.train()
    loader = get_loader(labeled_indices)
    for _ in range(3):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = T(x)
            loss = ce(pred, y)
            opt_T.zero_grad(); loss.backward(); opt_T.step()


# In[13]:


def evaluate():
    T.eval()
    correct, total = 0, 0
    
    # Class-wise accuracy tracking
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = T(x)
            preds = logits.argmax(dim=1)
            
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            # Track per-class accuracy
            for i in range(len(y)):
                label = y[i].item()
                if label not in class_correct:
                    class_correct[label] = 0
                    class_total[label] = 0
                
                class_total[label] += 1
                if preds[i] == y[i]:
                    class_correct[label] += 1
    
    # Overall accuracy
    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")
    
    # Print per-class accuracy if we have all 10 classes
    if len(class_total) == 10:
        print("Per-class accuracy:")
        for label in sorted(class_total.keys()):
            class_acc = class_correct[label] / max(class_total[label], 1)
            print(f"  Class {label}: {class_acc:.4f} ({class_correct[label]}/{class_total[label]})")
    
    return acc


# In[14]:


# Main training loop with all three improvements
acc_list = []
label_counts = []
class_balance_stats = []

num_rounds = 5
for round in range(num_rounds):
    print(f"\n=== Active Learning Round {round+1} ===")
    
    # Using improvement 2: progressive image generation
    epoch_fraction = round / num_rounds
    train_gan(epoch=1, current_epoch=round, total_epochs=num_rounds)
    
    # Using improvement 1: adaptive sampling
    selected_indices, selected_images, selected_labels = sample_from_pool(500, epoch_fraction)
    
    # Using improvement 3: class-balanced generation
    noise_factor = min(0.05 + (epoch_fraction * 0.15), 0.2)  # Match the noise schedule from train_gan
    gen_images, gen_labels = generate_balanced_samples(selected_images, selected_labels, noise_factor)
    
    # Update the labeled and unlabeled sets
    labeled_indices = np.concatenate([labeled_indices, selected_indices])
    unlabeled_indices = np.setdiff1d(unlabeled_indices, selected_indices)
    
    if gen_images is not None and len(gen_images) > 0:
        with torch.no_grad():
            # Save the first 25 generated samples for visualization
            gen_folder = f"round_{round+1}_generated"
            os.makedirs(gen_folder, exist_ok=True)
            if len(gen_images) >= 25:
                vutils.save_image(gen_images[:25], f"{gen_folder}/class_balanced_gen.png", 
                                  nrow=5, normalize=True)
                print(f"Saved class-balanced generated images to {gen_folder}/class_balanced_gen.png")
    
    train_T()
    acc = evaluate()
    acc_list.append(acc)
    label_counts.append(len(labeled_indices))
    
    # Track class balance stats
    class_counts = {}
    for idx in labeled_indices:
        label = dataset[idx][1]
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    
    class_balance_stats.append(class_counts)


# In[15]:


# Plotting results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(label_counts, acc_list, marker='o')
plt.xlabel("Number of Labeled Samples")
plt.ylabel("Test Accuracy")
plt.title("DGAAL Accuracy vs Labeled Data")
plt.grid(True)

# Plot class distribution evolution
plt.subplot(1, 2, 2)
class_labels = list(range(10))  # CIFAR10 has 10 classes
x = list(range(len(class_balance_stats)))

for cls in class_labels:
    y = [stats.get(cls, 0) for stats in class_balance_stats]
    plt.plot(x, y, marker='o', label=f'Class {cls}')

plt.xlabel("Round")
plt.ylabel("Number of Samples")
plt.title("Class Distribution Evolution")
plt.legend()
plt.tight_layout()
plt.show()


# In[16]:


# Visual sample generation
new_samples, xU_batch, _ = sample_from_pool(500)

# Save visual samples
import os, torchvision.utils as vutils

round_folder = f"round_{round+1}"
os.makedirs(round_folder, exist_ok=True)

# Save real
vutils.save_image(xU_batch[:25], f"{round_folder}/new2_sampled_real_round{round+1}.png", nrow=5, normalize=True)

# Save generated
z = g1(xU_batch[:25].to(DEVICE))
# Apply final noise level to showcase the progressive generation
noise_factor = 0.2
noise = torch.randn_like(z) * noise_factor
z_noisy = z + noise
xg = g2(z_noisy)
vutils.save_image(xg, f"{round_folder}/new2_sampled_generated_round{round+1}.png", nrow=5, normalize=True)

print(f"[✓] Saved sampled_real_round{round+1}.png and sampled_generated_round{round+1}.png")


# In[ ]:




