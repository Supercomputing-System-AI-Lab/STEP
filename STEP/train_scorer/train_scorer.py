import os
import copy
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_TRAIN_DIR = '/work/nvme/bcjw/zliang2/hidden_state_results/1016'

# Feature extraction settings
FEATURE_MODE = "last"       # Options: "last", "layer_mean", "concat", "certain_layer"
NUM_LAYERS_TO_USE = None    # Number of layers for "layer_mean" or layer index for "certain_layer"

# Training settings
VAL_SIZE = 0.20
RANDOM_STATE = 0
BATCH_SIZE = 128
MAX_EPOCHS = 20
PATIENCE = 5                # Early stopping patience

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MLP classifier on hidden states"
    )
    parser.add_argument(
        '--train_dir', 
        type=str, 
        default=DEFAULT_TRAIN_DIR,
        help='Directory containing training data (.pt files)'
    )
    parser.add_argument(
        '--step_sample_ratio', 
        type=float, 
        default=1,
        help='Ratio of timesteps to sample from each example (default: 1 = use all timesteps)'
    )
    parser.add_argument(
        '--config_name', 
        type=str, 
        default='default_config',
        help='Configuration name for saving models and plots'
    )
    return parser.parse_args()


# ============================================================================
# Model Definition
# ============================================================================

class MLPClassifier(nn.Module):
    """
    Simple MLP classifier for binary classification of hidden states.
    
    Architecture:
        Input -> Linear(512) -> ReLU -> Linear(1) -> Output (logits)
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        print(f"Initializing MLP with input dimension: {input_dim}")
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# Data Loading and Feature Extraction
# ============================================================================

def load_hidden_states(file_path: str, step_sample_ratio: float = 0.2):
    """
    Load hidden states from .pt files and sample timesteps.
    
    Args:
        file_path: Directory containing .pt files with hidden states
        step_sample_ratio: Fraction of timesteps to randomly sample per example
        
    Returns:
        X: numpy array of shape [N, L, D] where N=samples, L=layers, D=hidden_dim
        y: numpy array of labels (0 or 1)
    """
    all_hidden_states = []
    all_labels = []

    if not os.path.exists(file_path):
        print(f"Path not found: {file_path}")
        return np.array([]), np.array([])

    filenames = sorted([f for f in os.listdir(file_path) if f.endswith('.pt')])
    
    for filename in tqdm(filenames, desc=f"Loading from {file_path}"):
        full_path = os.path.join(file_path, filename)
        try:
            data = torch.load(full_path, map_location='cpu')
            if not isinstance(data, list):
                continue

            for item in data:
                if 'select_hidden_states' not in item:
                    continue

                hidden_state = item['select_hidden_states']
                if hidden_state.is_cuda:
                    hidden_state = hidden_state.cpu()

                # Ensure shape is [T, L, D] where T=timesteps, L=layers, D=hidden_dim
                if hidden_state.dim() == 2:
                    hidden_state = hidden_state.unsqueeze(1)  # [T, 1, D]
                else:
                    hidden_state = hidden_state.permute(1, 0, 2)  # [T, L, D]
                    
                # Sample timesteps
                num_timesteps = hidden_state.shape[0]
                num_samples = max(1, int(num_timesteps * step_sample_ratio))
                sampled_indices = np.random.choice(
                    num_timesteps, size=num_samples, replace=False
                )
                
                # Extract label
                label = 1 if item.get('is_correct', False) else 0
                
                for idx in sampled_indices:
                    hidden_state_step = hidden_state[idx].detach().numpy()  # [L, D]
                    all_hidden_states.append(hidden_state_step)
                    all_labels.append(label)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    if len(all_hidden_states) == 0:
        return np.array([]), np.array([])

    X = np.stack(all_hidden_states, axis=0)
    y = np.asarray(all_labels, dtype=np.int64)
    return X, y


def build_features(X: np.ndarray, mode: str = "layer_mean", num_layers: int = None) -> np.ndarray:
    """
    Extract features from multi-layer hidden states.
    
    Args:
        X: Input array of shape [N, L, D]
        mode: Feature extraction mode
            - "last": Use only the last layer
            - "layer_mean": Average over specified number of last layers
            - "concat": Concatenate all layers
            - "certain_layer": Use a specific layer by index
        num_layers: Number of layers (for "layer_mean") or layer index (for "certain_layer")
        
    Returns:
        Features array of shape [N, feature_dim]
    """
    if mode == "layer_mean":
        if num_layers is None:
            return X.mean(axis=1)
        layer_indices = list(range(X.shape[1] - num_layers, X.shape[1]))
        return X[:, layer_indices, :].mean(axis=1)
    
    elif mode == "concat":
        return X.reshape(X.shape[0], -1)
    
    elif mode == "last":
        return X[:, -1, :]
    
    elif mode == "certain_layer":
        if num_layers is None:
            raise ValueError("num_layers must be specified for 'certain_layer' mode")
        return X[:, num_layers, :]
    
    else:
        raise ValueError(f"Unknown feature extraction mode: {mode}")


# ============================================================================
# Evaluation Metrics
# ============================================================================

def evaluate_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, prefix: str = "") -> dict:
    """
    Compute and print classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred_prob: Predicted probabilities
        prefix: Prefix string for printing
        
    Returns:
        Dictionary containing AUC, accuracy, F1, precision, and recall
    """
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        auc = 0.5  # Default when only one class present

    print(f"\n{prefix}Results:")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}\n")
    
    return dict(auc=auc, acc=acc, f1=f1, precision=prec, recall=rec)


# ============================================================================
# Training Loop
# ============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config_name: str,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    checkpoint_dir: str = ''
) -> tuple:
    """
    Train the model with early stopping based on validation performance.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config_name: Name for logging and checkpointing
        lr: Learning rate
        weight_decay: L2 regularization weight
        checkpoint_dir: Directory to save model checkpoints
        
    Returns:
        Tuple of (loss_curve, validation_metrics)
    """
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    model = model.to(DEVICE)
    
    # Compute class weights for imbalanced data
    num_pos = sum((y == 1).sum().item() for _, y in train_loader)
    num_neg = sum((y == 0).sum().item() for _, y in train_loader)
    
    if num_pos > 0:
        pos_weight = torch.tensor([num_neg / num_pos]).to(DEVICE)
        print(f"Class distribution - Positive: {num_pos}, Negative: {num_neg}")
        print(f"Using pos_weight: {pos_weight.item():.4f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        print("Warning: No positive samples found, using standard BCEWithLogitsLoss")
        criterion = nn.BCEWithLogitsLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    loss_curve = []
    best_val_score = float('-inf')
    best_model_state = None
    patience_counter = 0
    
    print(f"\nTraining '{config_name}' on {DEVICE}...")
    print("-" * 60)
    
    for epoch in range(MAX_EPOCHS):
        # Training phase
        model.train()
        epoch_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        loss_curve.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_probs = []
        val_targets = []
        
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                y_val_cpu = y_val_batch.numpy()
                
                X_val_batch = X_val_batch.to(DEVICE)
                y_val_batch_gpu = y_val_batch.to(DEVICE).float().unsqueeze(1)
                
                outputs = model(X_val_batch)
                loss = criterion(outputs, y_val_batch_gpu)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                val_probs.extend(probs.cpu().numpy())
                val_targets.extend(y_val_cpu)
        
        avg_val_loss = val_loss / len(val_loader)

        val_probs = np.array(val_probs).flatten()
        val_targets = np.array(val_targets).flatten()

        # Compute validation metrics
        current_metrics = evaluate_metrics(
            val_targets, val_probs,
            prefix=f"[{config_name}] Epoch {epoch+1} "
        )
        
        print(f"[{config_name}] Epoch [{epoch+1}/{MAX_EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping based on combined score (weighted AUC + F1)
        current_auc = current_metrics['auc']
        current_f1 = current_metrics['f1']
        current_score = 0.75 * current_auc + 0.25 * current_f1

        if current_score > best_val_score + 1e-4:  # Threshold to prevent oscillation
            best_val_score = current_score
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            # Save checkpoint
            if checkpoint_dir:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_score': best_val_score,
                    'metrics': current_metrics,
                    'config_name': config_name,
                    'loss_curve': loss_curve,
                }
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f'checkpoint_epoch{epoch+1}_auc{current_auc:.4f}_f1{current_f1:.4f}.pt'
                )
                torch.save(checkpoint, checkpoint_path)
                print(f"  -> Checkpoint saved: {checkpoint_path}")

            print(f"  -> New best score: {current_score:.4f} "
                  f"(AUC: {current_auc:.4f}, F1: {current_f1:.4f})")
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
            
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\nTraining of '{config_name}' completed.")
    print("Running final evaluation on validation set...")
    
    # Final evaluation with best model
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch = X_val_batch.to(DEVICE)
            outputs = model(X_val_batch)
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(y_val_batch.numpy())
            
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    val_metrics = evaluate_metrics(
        all_targets, all_preds,
        prefix=f"[{config_name}] Final Validation "
    )
    
    return loss_curve, val_metrics


# ============================================================================
# Visualization
# ============================================================================

def plot_loss_curves(results_dict: dict, output_dir: str = './plots'):
    """Plot training loss curves for all configurations."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(12, 8))
    for config_name, data in results_dict.items():
        if 'loss_curve' in data and len(data['loss_curve']) > 0:
            plt.plot(data['loss_curve'], label=config_name, linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (BCE)', fontsize=12)
    plt.title('Training Loss Curves', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=150)
    plt.close()
    print(f"Loss curves saved to {output_dir}/loss_curves.png")


def plot_validation_metrics(results_dict: dict, output_dir: str = './plots'):
    """Plot bar charts of validation metrics for all configurations."""
    os.makedirs(output_dir, exist_ok=True)
    
    configs = []
    metrics_data = {'AUC': [], 'Accuracy': [], 'F1': []}
    
    for config_name, data in results_dict.items():
        if 'val_metrics' in data:
            configs.append(config_name)
            metrics = data['val_metrics']
            metrics_data['AUC'].append(metrics.get('auc', 0))
            metrics_data['Accuracy'].append(metrics.get('acc', 0))
            metrics_data['F1'].append(metrics.get('f1', 0))
    
    if not configs:
        print("No validation metrics to plot.")
        return
            
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = sns.color_palette("husl", len(configs))
    
    for idx, (metric_name, values) in enumerate(metrics_data.items()):
        ax = axes[idx]
        bars = ax.bar(range(len(configs)), values, color=colors)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.set_title(metric_name, fontsize=14)
        ax.set_ylim([0, 1.05])
        ax.set_ylabel('Score', fontsize=12)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_metrics.png'), dpi=150)
    plt.close()
    print(f"Validation metrics saved to {output_dir}/validation_metrics.png")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    # Load data
    print(f"Loading data from: {args.train_dir}")
    X_raw, y_raw = load_hidden_states(args.train_dir, step_sample_ratio=args.step_sample_ratio)
    
    if len(X_raw) == 0:
        print("No data found. Exiting.")
        return

    print(f"Total samples loaded: {X_raw.shape[0]}")
    print(f"Hidden state shape: {X_raw.shape}")  # [N, L, D]

    # Build features from hidden states
    features = build_features(X_raw, mode=FEATURE_MODE, num_layers=NUM_LAYERS_TO_USE)
    print(f"Feature shape after extraction: {features.shape}")

    # Split into train/validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        features, y_raw,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_raw
    )
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Training class distribution: {np.bincount(y_train)}")
    print(f"Validation class distribution: {np.bincount(y_val)}")

    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize and train model
    input_dim = X_train.shape[1]
    model = MLPClassifier(input_dim)
    
    checkpoint_dir = f'./checkpoints/{args.config_name}'
    loss_curve, val_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config_name=args.config_name,
        lr=1e-4,
        weight_decay=1e-5,
        checkpoint_dir=checkpoint_dir
    )
    
    # Store results
    all_results = {
        args.config_name: {
            'loss_curve': loss_curve,
            'val_metrics': val_metrics
        }
    }

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating visualization plots...")
    plot_dir = f'./plots/{args.config_name}'
    plot_loss_curves(all_results, output_dir=plot_dir)
    plot_validation_metrics(all_results, output_dir=plot_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for config, data in all_results.items():
        m = data['val_metrics']
        print(f"Config: {config}")
        print(f"  AUC:      {m['auc']:.4f}")
        print(f"  Accuracy: {m['acc']:.4f}")
        print(f"  F1:       {m['f1']:.4f}")
        print(f"  Precision:{m['precision']:.4f}")
        print(f"  Recall:   {m['recall']:.4f}")


if __name__ == "__main__":
    main()