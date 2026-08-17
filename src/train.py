import librosa_cache  # noqa: F401  -- must be imported before `librosa` itself (see librosa_cache.py)

import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import dataset
import model
import os

RATE = 48000
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8
SPLIT_SEED = 42  # fixed seed so the train/test song split is reproducible run-to-run

device = model.DEVICE

# Using relative paths from the workspace root "data/0001-1000-audio-mixes_2"
audio_path = os.path.join("data", "0001-1000-audio-mixes_2")
annot_path = os.path.join("data", "0001-1000-annotations-v1.1.0")

print(f"Audio Path: {audio_path}")
print(f"Annotation Path: {annot_path}")

# --- Split by SONG, before any windows/segments exist ---
all_files = sorted(glob.glob(os.path.join(audio_path, "*.flac")))
rng = random.Random(SPLIT_SEED)
shuffled_files = all_files[:]
rng.shuffle(shuffled_files)
split_idx = int(TRAIN_SPLIT * len(shuffled_files))
train_files, test_files = shuffled_files[:split_idx], shuffled_files[split_idx:]
print(f"Song-level split: {len(train_files)} train songs, {len(test_files)} test songs")

train_raw = dataset.ChordDataset(
    audio_dir=audio_path,
    annotation_dir=annot_path,
    sr=RATE,
    file_list=train_files,
)
test_raw = dataset.ChordDataset(
    audio_dir=audio_path,
    annotation_dir=annot_path,
    sr=RATE,
    file_list=test_files,
)

# load and preprocess data
print("Initializing Segmented Datasets (this might take a few moments)...")
train_data = model.ChordSegmentDataset(train_raw)
test_data = model.ChordSegmentDataset(test_raw)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

#init the model
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

cnn = model.ChordCNN1D(num_classes=model.NUM_CLASSES, input_bins=84).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

#resume from checkpoint if exists
start_epoch = 0
latest_checkpoint = os.path.join(checkpoint_dir, "latest.pth")
if os.path.exists(latest_checkpoint):
    print("Resuming from checkpoint...")
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    cnn.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from epoch {start_epoch}")

# training loop
print("Starting Training...")
for epoch in range(start_epoch, EPOCHS):
    cnn.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (spectograms, labels) in enumerate(train_loader):
        spectograms = spectograms.to(device)
        labels = labels.to(device)
        
        # forwrd
        outputs = cnn(spectograms)
        loss = criterion(outputs, labels)
        
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (i+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    avg_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

    # save checkpoint
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': cnn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'accuracy': train_acc,
    }
    torch.save(checkpoint_data, os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")) #save every epoch
    torch.save(checkpoint_data, latest_checkpoint) #overide latest
    print(f"Checkpoint saved for epoch {epoch+1}")


#EVALL
print("Evaluating on test set...")
cnn.eval()
test_correct = 0
test_total = 0
test_loss = 0.0

with torch.no_grad():
    for spectograms, labels in test_loader:
        spectograms = spectograms.to(device)
        labels = labels.to(device)
        
        outputs = cnn(spectograms)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

test_acc = 100 * test_correct / test_total
avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

print("Training complete. Final weights are in checkpoints/latest.pth")