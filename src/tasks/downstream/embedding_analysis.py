"""
Embedding Analysis Script for GENERanno

This script:
1. Extracts embeddings from a trained/pretrained model for sequences in CSV files
2. Trains a linear probe (logistic regression) classifier
3. Calculates silhouette score to measure embedding quality
4. Creates PCA visualization showing class separation
5. Trains a simple 3-layer neural network classifier

Usage:
    python -m src.tasks.downstream.embedding_analysis \
        --csv_dir /path/to/csv/data \
        --model_path /path/to/finetuned/model \
        --output_dir ./results/embedding_analysis
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from datasets import Dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract embeddings and perform embedding analysis"
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        required=True,
        help="Path to directory containing train.csv, dev.csv, test.csv",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="GenerTeam/GENERanno-prokaryote-0.5b-base",
        help="Path to trained model or HuggingFace model name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/embedding_analysis",
        help="Directory to save results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for embedding extraction",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "cls", "last"],
        help="Pooling strategy for embeddings (mean, cls, last)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--nn_epochs",
        type=int,
        default=100,
        help="Number of epochs for 3-layer NN training",
    )
    parser.add_argument(
        "--nn_hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for 3-layer NN",
    )
    parser.add_argument(
        "--nn_lr",
        type=float,
        default=1e-3,
        help="Learning rate for 3-layer NN",
    )
    return parser.parse_args()


def load_csv_data(csv_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test CSV files."""
    train_path = os.path.join(csv_dir, "train.csv")
    test_path = os.path.join(csv_dir, "test.csv")

    # Check for dev.csv or val.csv
    dev_path = os.path.join(csv_dir, "dev.csv")
    val_path = os.path.join(csv_dir, "val.csv")
    if os.path.exists(dev_path):
        validation_path = dev_path
    elif os.path.exists(val_path):
        validation_path = val_path
    else:
        raise FileNotFoundError(f"No validation file found in {csv_dir}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    print(f"Loaded data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def extract_embeddings(
    model,
    tokenizer,
    sequences: List[str],
    labels: List[int],
    batch_size: int,
    max_length: int,
    pooling: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings from model for given sequences.

    Args:
        model: The transformer model
        tokenizer: The tokenizer
        sequences: List of DNA sequences
        labels: List of labels
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        pooling: Pooling strategy ('mean', 'cls', 'last')
        device: Device to run on

    Returns:
        Tuple of (embeddings array, labels array)
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    # Process in batches
    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
        batch_seqs = sequences[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_seqs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

            # Get last hidden state
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states[-1]
            else:
                raise ValueError("Cannot extract hidden states from model output")

            # Apply pooling
            attention_mask = inputs.get('attention_mask', None)

            if pooling == "mean":
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                else:
                    embeddings = hidden_states.mean(dim=1)
            elif pooling == "cls":
                embeddings = hidden_states[:, 0, :]
            elif pooling == "last":
                if attention_mask is not None:
                    # Get the last non-padded token
                    seq_lengths = attention_mask.sum(dim=1) - 1
                    embeddings = hidden_states[torch.arange(hidden_states.size(0)), seq_lengths]
                else:
                    embeddings = hidden_states[:, -1, :]

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(batch_labels)

    embeddings_array = np.vstack(all_embeddings)
    labels_array = np.array(all_labels)

    return embeddings_array, labels_array


def train_linear_probe(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
) -> Dict[str, float]:
    """
    Train a linear probe (logistic regression) classifier.

    Returns:
        Dictionary of metrics
    """
    print("\n" + "=" * 60)
    print("Training Linear Probe (Logistic Regression)")
    print("=" * 60)

    # Standardize embeddings
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_embeddings)
    test_scaled = scaler.transform(test_embeddings)

    # Train logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        random_state=seed,
        solver='lbfgs',
        n_jobs=-1,
    )
    clf.fit(train_scaled, train_labels)

    # Predict
    test_preds = clf.predict(test_scaled)
    test_probs = clf.predict_proba(test_scaled)[:, 1]

    # Calculate metrics
    metrics = {
        "linear_probe_accuracy": float(accuracy_score(test_labels, test_preds)),
        "linear_probe_precision": float(precision_score(test_labels, test_preds, zero_division=0)),
        "linear_probe_recall": float(recall_score(test_labels, test_preds, zero_division=0)),
        "linear_probe_f1": float(f1_score(test_labels, test_preds, zero_division=0)),
        "linear_probe_mcc": float(matthews_corrcoef(test_labels, test_preds)),
    }

    # Add AUC if binary
    try:
        metrics["linear_probe_auc"] = float(roc_auc_score(test_labels, test_probs))
    except ValueError:
        metrics["linear_probe_auc"] = 0.0

    # Sensitivity and Specificity
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds, labels=[0, 1]).ravel()
    metrics["linear_probe_sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["linear_probe_specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    print(f"  Accuracy: {metrics['linear_probe_accuracy']:.4f}")
    print(f"  F1 Score: {metrics['linear_probe_f1']:.4f}")
    print(f"  MCC: {metrics['linear_probe_mcc']:.4f}")
    print(f"  AUC: {metrics['linear_probe_auc']:.4f}")

    return metrics


def calculate_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Calculate silhouette score for embeddings.

    Higher scores indicate better-defined clusters.
    Range: [-1, 1], where 1 is best.
    """
    print("\n" + "=" * 60)
    print("Calculating Silhouette Score")
    print("=" * 60)

    # Standardize for fair comparison
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    # Calculate silhouette score
    score = silhouette_score(scaled_embeddings, labels)
    print(f"  Silhouette Score: {score:.4f}")
    print(f"  Interpretation: ", end="")
    if score > 0.5:
        print("Strong structure (embeddings well-separated by class)")
    elif score > 0.25:
        print("Reasonable structure")
    elif score > 0:
        print("Weak structure (some overlap between classes)")
    else:
        print("No apparent structure (classes highly overlapped)")

    return float(score)


def create_pca_visualization(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str = "PCA Visualization of Embeddings",
) -> Dict[str, float]:
    """
    Create PCA visualization of embeddings colored by class.

    Returns:
        Dictionary with explained variance ratios
    """
    print("\n" + "=" * 60)
    print("Creating PCA Visualization")
    print("=" * 60)

    # Standardize embeddings
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    # Fit PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(scaled_embeddings)

    explained_var = pca.explained_variance_ratio_
    print(f"  PC1 explains {explained_var[0]*100:.2f}% of variance")
    print(f"  PC2 explains {explained_var[1]*100:.2f}% of variance")
    print(f"  Total: {sum(explained_var)*100:.2f}%")

    # Create plot
    plt.figure(figsize=(10, 8))

    # Plot each class with different colors
    colors = ['#1f77b4', '#ff7f0e']  # Blue for 0, Orange for 1
    class_names = ['Class 0', 'Class 1']

    for class_idx in [0, 1]:
        mask = labels == class_idx
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors[class_idx],
            label=f'{class_names[class_idx]} (n={mask.sum()})',
            alpha=0.6,
            s=30,
        )

    plt.xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved to: {output_path}")

    return {
        "pca_explained_variance_pc1": float(explained_var[0]),
        "pca_explained_variance_pc2": float(explained_var[1]),
        "pca_total_explained_variance": float(sum(explained_var)),
    }


class ThreeLayerNN(nn.Module):
    """Simple 3-layer neural network for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.network(x)


def train_three_layer_nn(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    hidden_dim: int,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> Tuple[Dict[str, float], nn.Module, StandardScaler]:
    """
    Train a 3-layer neural network classifier on embeddings.

    Returns:
        Tuple of (metrics dict, trained model, scaler)
    """
    print("\n" + "=" * 60)
    print("Training 3-Layer Neural Network")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Standardize embeddings
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_embeddings)
    val_scaled = scaler.transform(val_embeddings)
    test_scaled = scaler.transform(test_embeddings)

    # Create tensors
    train_X = torch.FloatTensor(train_scaled).to(device)
    train_y = torch.LongTensor(train_labels).to(device)
    val_X = torch.FloatTensor(val_scaled).to(device)
    val_y = torch.LongTensor(val_labels).to(device)
    test_X = torch.FloatTensor(test_scaled).to(device)
    test_y = torch.LongTensor(test_labels).to(device)

    # Create data loaders
    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize model
    input_dim = train_embeddings.shape[1]
    model = ThreeLayerNN(input_dim, hidden_dim).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=False
    )

    # Training loop
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0
    patience = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_X)
            val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            val_f1 = f1_score(val_labels, val_preds, zero_division=0)

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}, Val F1: {val_f1:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Test evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_X)
        test_probs = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
        test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()

    # Calculate metrics
    metrics = {
        "nn_accuracy": float(accuracy_score(test_labels, test_preds)),
        "nn_precision": float(precision_score(test_labels, test_preds, zero_division=0)),
        "nn_recall": float(recall_score(test_labels, test_preds, zero_division=0)),
        "nn_f1": float(f1_score(test_labels, test_preds, zero_division=0)),
        "nn_mcc": float(matthews_corrcoef(test_labels, test_preds)),
    }

    try:
        metrics["nn_auc"] = float(roc_auc_score(test_labels, test_probs))
    except ValueError:
        metrics["nn_auc"] = 0.0

    # Sensitivity and Specificity
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds, labels=[0, 1]).ravel()
    metrics["nn_sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["nn_specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    print(f"\n  Final Test Results:")
    print(f"  Accuracy: {metrics['nn_accuracy']:.4f}")
    print(f"  F1 Score: {metrics['nn_f1']:.4f}")
    print(f"  MCC: {metrics['nn_mcc']:.4f}")
    print(f"  AUC: {metrics['nn_auc']:.4f}")

    return metrics, model, scaler


def main():
    """Main function to run embedding analysis."""
    args = parse_arguments()

    print("\n" + "=" * 60)
    print("GENERanno Embedding Analysis")
    print("=" * 60)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_df, val_df, test_df = load_csv_data(args.csv_dir)

    # Load model and tokenizer
    print(f"\nLoading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    model = model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Extract embeddings
    print("\nExtracting train embeddings...")
    train_embeddings, train_labels = extract_embeddings(
        model, tokenizer,
        train_df["sequence"].tolist(),
        train_df["label"].tolist(),
        args.batch_size, args.max_length, args.pooling, device,
    )

    print("\nExtracting validation embeddings...")
    val_embeddings, val_labels = extract_embeddings(
        model, tokenizer,
        val_df["sequence"].tolist(),
        val_df["label"].tolist(),
        args.batch_size, args.max_length, args.pooling, device,
    )

    print("\nExtracting test embeddings...")
    test_embeddings, test_labels = extract_embeddings(
        model, tokenizer,
        test_df["sequence"].tolist(),
        test_df["label"].tolist(),
        args.batch_size, args.max_length, args.pooling, device,
    )

    print(f"\nEmbedding shape: {test_embeddings.shape}")

    # Save embeddings
    embeddings_path = os.path.join(args.output_dir, "embeddings.npz")
    np.savez(
        embeddings_path,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
    )
    print(f"\nSaved embeddings to: {embeddings_path}")

    # Initialize results dictionary
    results = {
        "model_path": args.model_path,
        "csv_dir": args.csv_dir,
        "pooling": args.pooling,
        "embedding_dim": int(test_embeddings.shape[1]),
        "train_samples": len(train_labels),
        "val_samples": len(val_labels),
        "test_samples": len(test_labels),
    }

    # 1. Train linear probe
    linear_metrics = train_linear_probe(
        train_embeddings, train_labels,
        test_embeddings, test_labels,
        args.seed,
    )
    results.update(linear_metrics)

    # 2. Calculate silhouette score
    silhouette = calculate_silhouette(test_embeddings, test_labels)
    results["silhouette_score"] = silhouette

    # 3. Create PCA visualization
    pca_path = os.path.join(args.output_dir, "pca_visualization.png")
    pca_metrics = create_pca_visualization(
        test_embeddings, test_labels,
        pca_path,
        title=f"PCA of Test Embeddings\n(Silhouette: {silhouette:.3f})",
    )
    results.update(pca_metrics)

    # 4. Train 3-layer NN
    nn_metrics, nn_model, nn_scaler = train_three_layer_nn(
        train_embeddings, train_labels,
        val_embeddings, val_labels,
        test_embeddings, test_labels,
        args.nn_hidden_dim, args.nn_epochs, args.nn_lr,
        args.seed, device,
    )
    results.update(nn_metrics)

    # Save NN model
    nn_model_path = os.path.join(args.output_dir, "three_layer_nn.pt")
    torch.save({
        "model_state_dict": nn_model.state_dict(),
        "input_dim": test_embeddings.shape[1],
        "hidden_dim": args.nn_hidden_dim,
    }, nn_model_path)
    print(f"\nSaved 3-layer NN to: {nn_model_path}")

    # Save results
    results_path = os.path.join(args.output_dir, "embedding_analysis_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nLinear Probe Results:")
    print(f"  Accuracy: {results['linear_probe_accuracy']:.4f}")
    print(f"  MCC: {results['linear_probe_mcc']:.4f}")
    print(f"  AUC: {results['linear_probe_auc']:.4f}")
    print(f"\n3-Layer NN Results:")
    print(f"  Accuracy: {results['nn_accuracy']:.4f}")
    print(f"  MCC: {results['nn_mcc']:.4f}")
    print(f"  AUC: {results['nn_auc']:.4f}")
    print(f"\nEmbedding Quality:")
    print(f"  Silhouette Score: {results['silhouette_score']:.4f}")
    print(f"  PCA Variance Explained: {results['pca_total_explained_variance']*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
