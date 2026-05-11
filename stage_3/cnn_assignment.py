"""
Thomas Watson (claude did most of this lol)
05/10/2026
Project Stage 3: CNN Assignment - MNIST, ORL, and CIFAR-10
==========================================

Requirements:
    pip install torch torchvision scikit-learn matplotlib numpy
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────
# SECTION 1: CUSTOM DATASET CLASS
# ─────────────────────────────────────────────
# PyTorch needs data in a special "Dataset" wrapper so the DataLoader
# can batch and shuffle it automatically during training.

# so yeah this packs stuff up
class ImageDataset(Dataset):
    """
    A generic dataset class that wraps our pickled data dictionaries.

    Args:
        instances : list of {'image': np.array, 'label': int}
        channel   : 'single' → use only the R channel (for grayscale stored as 3-ch)
                    'multi'  → use all 3 channels (for colour images)
        img_size  : (H, W) to which every image will be resized, or None to keep as-is
    """

    def __init__(self, instances, channel='single', img_size=None):
        self.data = []

        for inst in instances:
            img   = np.array(inst['image'], dtype=np.float32)
            label = int(inst['label'])

            # ── Normalise pixel values to [0, 1] ──────────────────────────
            img = img / 255.0

            # ── Handle channel dimension ───────────────────────────────────
            if img.ndim == 2:
                # Shape (H, W)  →  (1, H, W)
                img = img[np.newaxis, :, :]

            elif img.ndim == 3 and img.shape[2] in (1, 3):
                # Shape (H, W, C)  →  (C, H, W)  [PyTorch convention]
                # claude is kinda crazy lmao
                img = img.transpose(2, 0, 1)

                if channel == 'single':
                    # Keep only the first channel (R), shape becomes (1, H, W)
                    img = img[0:1, :, :]

            # ── Optional resize (simple centre-crop / pad is avoided; ─────
            #    instead we just let the CNN handle the native size)   ─────

            self.data.append((
                torch.tensor(img, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ─────────────────────────────────────────────
# SECTION 2: CNN MODEL DEFINITION
# ─────────────────────────────────────────────

class CNN(nn.Module):
    """
    Configurable Convolutional Neural Network.

    Parameters you can tweak for task 3-5
    ──────────────────────────────────────
    in_channels      : number of input channels (1 for grayscale, 3 for colour)
    num_classes      : how many output categories
    conv_configs     : list of dicts, one per conv block:
                         out_channels  – number of filters
                         kernel_size   – size of each filter (e.g. 3 → 3×3)
                         stride        – step size of the filter (default 1)
                         padding       – zero-padding around the image (default 1)
                         pool          – True/False, add a 2×2 max-pool after activation
    hidden_dim       : width of the fully-connected hidden layer
    dropout_rate     : fraction of neurons randomly zeroed during training (regularisation)
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        conv_configs=None,
        hidden_dim=256,
        dropout_rate=0.5,
    ):
        super().__init__()

        # Default architecture if none supplied
        if conv_configs is None:
            conv_configs = [
                {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': True},
                {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': True},
            ]

        # ── Build conv blocks dynamically ────────────────────────────────
        layers = []
        current_channels = in_channels

        for cfg in conv_configs:
            layers += [
                nn.Conv2d(
                    in_channels  = current_channels,
                    out_channels = cfg['out_channels'],
                    kernel_size  = cfg.get('kernel_size', 3),
                    stride       = cfg.get('stride', 1),
                    padding      = cfg.get('padding', 1),
                ),
                nn.BatchNorm2d(cfg['out_channels']),   # stabilises training
                nn.ReLU(inplace=True),
            ]
            if cfg.get('pool', True):
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            current_channels = cfg['out_channels']

        self.conv_net = nn.Sequential(*layers)

        # ── Adaptive pooling squeezes spatial dims to a fixed 4×4 ────────
        # This means the FC layer size doesn't change even if input
        # images have different resolutions.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        flat_size = current_channels * 4 * 4

        # ── Fully-connected classification head ──────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x   # raw logits (no softmax — CrossEntropyLoss handles that)


# ─────────────────────────────────────────────
# SECTION 3: TRAINING & EVALUATION HELPERS
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one full pass over the training set. Returns avg loss and accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # clear gradients from previous step
        outputs = model(images)        # forward pass
        loss    = criterion(outputs, labels)
        loss.backward()                # compute gradients
        optimizer.step()               # update weights

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model on a DataLoader. Returns avg loss, accuracy, and all predictions."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)
            all_preds  .extend(preds.cpu().numpy())
            all_labels .extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=20,
    lr=1e-3,
    device=None,
    loss_fn='cross_entropy',   # 'cross_entropy' or 'nll'
):
    """
    Full training loop.

    Returns
    -------
    history : dict with lists 'train_loss', 'train_acc', 'test_loss', 'test_acc'
    """

    # check for cuda
    # turns out my computer has cuda... wow this makes it so much faster omg
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ── Loss function ───────────────────────
    if loss_fn == 'nll':
        criterion = nn.NLLLoss()
        # NLLLoss expects log-probabilities → wrap model output with LogSoftmax
        model = nn.Sequential(model, nn.LogSoftmax(dim=1))
    else:
        criterion = nn.CrossEntropyLoss()   # standard; handles raw logits

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Reduce LR by ×0.5 if validation loss doesn't improve for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step(te_loss)

        history['train_loss'].append(tr_loss)
        history['train_acc'] .append(tr_acc)
        history['test_loss'] .append(te_loss)
        history['test_acc']  .append(te_acc)

        print(f"Epoch {epoch:>3}/{num_epochs} | "
              f"Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f} | "
              f"Test  loss: {te_loss:.4f}  acc: {te_acc:.4f}")

    return history, model, criterion, device


# ─────────────────────────────────────────────
# SECTION 4: LEARNING CURVE PLOTTING
# ─────────────────────────────────────────────

def plot_learning_curves(history, title, save_path=None):
    """
    Plot loss and accuracy curves for both train and test sets.
    Saves to file if save_path is provided.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=14)

    # Loss curve
    ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3)
    ax1.plot(epochs, history['test_loss'],  label='Test Loss',  marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Epochs')
    ax1.legend()
    ax1.grid(True)

    # Accuracy curve
    ax2.plot(epochs, history['train_acc'], label='Train Accuracy', marker='o', markersize=3)
    ax2.plot(epochs, history['test_acc'],  label='Test Accuracy',  marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.legend()
    ax2.grid(True)
    # this isn't good for MNIST, can I improve it...
    # ok this is really stupid, BUT...
    # for MNIST, comment out this line
    #ax2.set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved learning curve → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# SECTION 5: DATASET LOADING UTILITY
# ─────────────────────────────────────────────

def load_dataset(path):
    """Load a pickle file and return the dict with 'train' and 'test' lists."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"  Loaded '{path}': {len(data['train'])} train, {len(data['test'])} test instances")
    return data


# ─────────────────────────────────────────────
# SECTION 6: MAIN EXPERIMENT RUNNER
# ─────────────────────────────────────────────

def run_experiment(
    dataset_path,
    dataset_name,
    in_channels,
    num_classes,
    channel_mode,
    conv_configs=None,
    hidden_dim=256,
    dropout_rate=0.5,
    batch_size=64,
    num_epochs=20,
    lr=1e-3,
    loss_fn='cross_entropy',
    label_offset=0,    # ORL labels start at 1; set offset=-1 to shift to 0-based
):
    """
    End-to-end: load data → build model → train → evaluate → plot curves.

    label_offset: added to every label; use -1 when labels are 1-indexed.
    """
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {dataset_name}")
    print(f"{'='*60}")

    # 1. Load raw data
    data = load_dataset(dataset_path)

    # 2. Adjust labels if needed (CrossEntropyLoss requires 0-based labels)
    if label_offset != 0:
        for split in ('train', 'test'):
            for inst in data[split]:
                inst['label'] = inst['label'] + label_offset

    # 3. Build PyTorch datasets and loaders
    train_ds = ImageDataset(data['train'], channel=channel_mode)
    test_ds  = ImageDataset(data['test'],  channel=channel_mode)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Build model
    model = CNN(
        in_channels  = in_channels,
        num_classes  = num_classes,
        conv_configs = conv_configs,
        hidden_dim   = hidden_dim,
        dropout_rate = dropout_rate,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model trainable parameters: {total_params:,}")

    # 5. Train
    history, trained_model, criterion, device = train_model(
        model, train_loader, test_loader,
        num_epochs=num_epochs, lr=lr, loss_fn=loss_fn,
    )

    # 6. Final evaluation
    _, final_acc, preds, true_labels = evaluate(trained_model, test_loader, criterion, device)
    print(f"\n  Final Test Accuracy: {final_acc:.4f} ({final_acc*100:.1f}%)")
    print("\n  Classification Report:")
    print(classification_report(true_labels, preds))

    # 7. Learning curves
    plot_learning_curves(history, title=f"{dataset_name} – Learning Curves",
                         save_path=f"{dataset_name.replace(' ', '_')}_curves.png")

    return history, final_acc


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: RUN ALL THREE DATASETS  (Tasks 3-3 and 3-4)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Task 3-3: MNIST ──────────────────────────────────────────────────
    # Grayscale 28×28, single channel, 10 digit classes (0-9)
    run_experiment(
        dataset_path  = 'MNIST',
        dataset_name  = 'MNIST Digits ',
        in_channels   = 1,
        num_classes   = 10,
        channel_mode  = 'single',
        batch_size    = 128,
        num_epochs    = 20,
        lr            = 1e-3,
    )

    # ── Task 3-4a: ORL Faces ─────────────────────────────────────────────
    # Stored as 112×92×3 but all channels equal (grayscale).
    # Labels are 1-40 → shift by -1 so they become 0-39.
    run_experiment(
        dataset_path  = 'ORL',
        dataset_name  = 'ORL Faces',
        in_channels   = 1,          # only use R channel
        num_classes   = 40,
        channel_mode  = 'single',
        batch_size    = 8,         # small dataset → small batch
        num_epochs    = 40,         # more epochs because fewer samples
        lr            = 5e-4,
        label_offset  = -1,         # shift 1-40 → 0-39
    )

    # ── Task 3-4b: CIFAR-10 ──────────────────────────────────────────────
    # Colour 32×32×3, 10 object classes (0-9)
    #run_experiment(
    #    dataset_path  = 'CIFAR',
    #    dataset_name  = 'CIFAR-10 Objects',
    #    in_channels   = 3,
    #    num_classes   = 10,
    #    channel_mode  = 'multi',
    #    batch_size    = 128,
    #    num_epochs    = 90,
    #    lr            = 1e-3,
    #)