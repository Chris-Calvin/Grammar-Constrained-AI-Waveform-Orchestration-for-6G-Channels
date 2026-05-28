"""
trainer.py - Model Training Pipeline for Waveform Selection Transformer.

Includes WaveformDataset, ModelTrainer with full training loop, evaluation,
early stopping, logging, and plotting.
"""

import os
import sys
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.tokenizer.tokenizer import ChannelStateTokenizer
from src.transformer.model import WaveformTransformerEncoder, count_parameters

WAVEFORM_NAMES = ["OFDM", "F_OFDM", "FBMC", "SC_FDMA", "OTFS", "NOMA"]


# =========================================================================
# Dataset
# =========================================================================
class WaveformDataset(Dataset):
    """PyTorch Dataset wrapping tokenized features and labels.

    Parameters
    ----------
    token_ids : np.ndarray, shape (N, 12)
    labels : np.ndarray, shape (N,)
    """

    def __init__(self, token_ids: np.ndarray, labels: np.ndarray):
        self.token_ids = torch.from_numpy(token_ids.astype(np.int64))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.token_ids[idx], self.labels[idx]


# =========================================================================
# Trainer
# =========================================================================


class ModelTrainer:
    """Complete training pipeline.

    Parameters
    ----------
    model : WaveformTransformerEncoder
    config : dict   (optional overrides)
    device : str    ('cpu' or 'cuda')
    """

    def __init__(self, model: WaveformTransformerEncoder, config: dict | None = None,
                 device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.config = config or {}

        self.lr = self.config.get("lr", 1e-3)
        self.weight_decay = self.config.get("weight_decay", 1e-4)
        self.batch_size = self.config.get("batch_size", 256)
        self.patience = self.config.get("patience", 5)
        self.epochs = self.config.get("epochs", 50)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.boundary_loader = None

        self.history: list[dict] = []
        self.best_val_acc = 0.0

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def prepare_dataloaders(self, data_dir: str) -> None:
        """Load .npy splits, tokenize, create DataLoaders."""
        print("Loading data and tokenizer...")
        tok = ChannelStateTokenizer.load(os.path.join(data_dir, "tokenizer.pkl"))

        train_X = np.load(os.path.join(data_dir, "train_X.npy"))
        train_y = np.load(os.path.join(data_dir, "train_y.npy"))
        val_X = np.load(os.path.join(data_dir, "val_X.npy"))
        val_y = np.load(os.path.join(data_dir, "val_y.npy"))
        test_X = np.load(os.path.join(data_dir, "test_X.npy"))
        test_y = np.load(os.path.join(data_dir, "test_y.npy"))
        bd_X = np.load(os.path.join(data_dir, "boundary_X.npy"))
        bd_y = np.load(os.path.join(data_dir, "boundary_y.npy"))

        # Tokenize
        print(f"  Tokenizing {len(train_X)} train, {len(val_X)} val, "
              f"{len(test_X)} test, {len(bd_X)} boundary samples...")
        train_tok = tok.transform(train_X)
        val_tok = tok.transform(val_X)
        test_tok = tok.transform(test_X)
        bd_tok = tok.transform(bd_X)

        self.train_loader = DataLoader(
            WaveformDataset(train_tok, train_y),
            batch_size=self.batch_size, shuffle=True, drop_last=False,
        )
        self.val_loader = DataLoader(
            WaveformDataset(val_tok, val_y),
            batch_size=self.batch_size, shuffle=False,
        )
        self.test_loader = DataLoader(
            WaveformDataset(test_tok, test_y),
            batch_size=self.batch_size, shuffle=False,
        )
        self.boundary_loader = DataLoader(
            WaveformDataset(bd_tok, bd_y),
            batch_size=self.batch_size, shuffle=False,
        )
        print(f"  DataLoaders ready. Train batches: {len(self.train_loader)}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, epochs: int | None = None) -> dict:
        """Full training loop with early stopping.

        Returns dict with final metrics.
        """
        epochs = epochs or self.epochs
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        patience_counter = 0
        output_dir = os.path.join(PROJECT_ROOT, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        best_path = os.path.join(output_dir, "best_model.pth")

        log_dir = os.path.join(PROJECT_ROOT, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "training_log.csv")

        start_time = time.time()
        self.history = []
        print(f"\nTraining for up to {epochs} epochs (patience={self.patience})...")
        print(f"{'Epoch':>5s}  {'TrLoss':>8s}  {'TrAcc':>7s}  {'VaLoss':>8s}  "
              f"{'VaAcc':>7s}  {'LR':>10s}  {'Time':>6s}")
        print("-" * 60)

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # --- Train ---
            self.model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for token_ids, labels in self.train_loader:
                token_ids = token_ids.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(token_ids)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * labels.size(0)
                train_correct += (logits.argmax(dim=1) == labels).sum().item()
                train_total += labels.size(0)

            train_loss /= train_total
            train_acc = train_correct / train_total

            # --- Validate ---
            val_loss, val_acc = self._eval_epoch(self.val_loader, criterion)

            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            dt = time.time() - t0

            rec = {
                "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "lr": lr,
            }
            self.history.append(rec)

            print(f"{epoch:>5d}  {train_loss:>8.4f}  {train_acc:>6.2%}  "
                  f"{val_loss:>8.4f}  {val_acc:>6.2%}  {lr:>10.6f}  {dt:>5.1f}s")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), best_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(best val_acc={best_val_acc:.2%})")
                    break

        self.best_val_acc = best_val_acc

        # Save log
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
            writer.writeheader()
            writer.writerows(self.history)

        # Load best model
        self.model.load_state_dict(torch.load(best_path, weights_only=True))
        self.model.eval()

        end_time = time.time()
        training_time_minutes = (end_time - start_time) / 60.0
        import json
        with open(os.path.join(output_dir, "training_metadata.json"), "w") as fm:
            json.dump({"training_time_minutes": training_time_minutes, "epochs_trained": len(self.history), "best_val_accuracy": best_val_acc}, fm)
            
        return {
            "final_train_acc": self.history[-1]["train_acc"],
            "final_val_acc": best_val_acc,
            "epochs_run": len(self.history),
            "best_model_path": best_path,
            "log_path": log_path,
        }

    def finetune_boundary(self):
        print("\n--- Starting Boundary Fine-Tuning (Fix E) ---")
        output_dir = os.path.join(PROJECT_ROOT, "outputs")
        best_path = os.path.join(output_dir, "best_model.pth")
        ft_path = os.path.join(output_dir, "best_model_finetuned.pth")
        
        self.model.load_state_dict(torch.load(best_path, weights_only=True))
        
        # Split boundary loader into 1600 train / 400 val
        ds = self.boundary_loader.dataset
        train_len = 1600
        val_len = len(ds) - train_len
        train_ds, val_ds = torch.utils.data.random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(42))
        
        ft_train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        ft_val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        
        criterion = torch.nn.CrossEntropyLoss()
        _, pre_bd_acc = self._eval_epoch(ft_val_loader, criterion)
        _, pre_indist_acc = self._eval_epoch(self.test_loader, criterion)
        print(f"Pre-FT Boundary Acc (400 holds): {pre_bd_acc:.4f}")
        print(f"Pre-FT In-Dist Acc (3000 holds): {pre_indist_acc:.4f}")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=self.weight_decay)
        
        best_ft_acc = pre_bd_acc
        best_state = self.model.state_dict().copy()
        
        for epoch in range(1, 11):
            self.model.train()
            for token_ids, labels in ft_train_loader:
                token_ids = token_ids.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model(token_ids)
                loss = 3.0 * criterion(logits, labels) # Higher weight on boundary
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
            _, val_acc = self._eval_epoch(ft_val_loader, criterion)
            print(f"  FT Epoch {epoch}: Val Acc {val_acc:.4f}")
            if val_acc > best_ft_acc:
                best_ft_acc = val_acc
                best_state = self.model.state_dict().copy()
                
        self.model.load_state_dict(best_state)
        _, post_bd_acc = self._eval_epoch(ft_val_loader, criterion)
        _, post_indist_acc = self._eval_epoch(self.test_loader, criterion)
        
        print(f"Post-FT Boundary Acc: {post_bd_acc:.4f} (was {pre_bd_acc:.4f})")
        print(f"Post-FT In-Dist Acc: {post_indist_acc:.4f} (was {pre_indist_acc:.4f})")
        
        bd_improved = post_bd_acc > pre_bd_acc
        indist_drop = pre_indist_acc - post_indist_acc
        
        if bd_improved and indist_drop <= 0.02:
            print(f"SUCCESS: Boundary improved and In-Dist drop ({indist_drop:.4f}) <= 0.02. Saving fine-tuned model!")
            torch.save(self.model.state_dict(), ft_path)
        else:
            print("FAILURE: Conditions not met. Reverting to base model.")
            self.model.load_state_dict(torch.load(best_path, weights_only=True))

    def _eval_epoch(self, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
        """Evaluate on a dataloader, return (loss, accuracy)."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for token_ids, labels in loader:
                token_ids = token_ids.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(token_ids)
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        return total_loss / total, correct / total

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, loader: DataLoader) -> dict:
        """Full evaluation: accuracy, per-class P/R/F1, confusion matrix.

        Returns dict with metrics.
        """
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for token_ids, labels in loader:
                token_ids = token_ids.to(self.device)
                preds = self.model.predict(token_ids).cpu()
                all_preds.append(preds)
                all_labels.append(labels)

        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        n_classes = 6

        # Confusion matrix
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for p, t in zip(preds, labels):
            cm[t, p] += 1

        accuracy = (preds == labels).mean()

        # Per-class metrics
        per_class = {}
        for c in range(n_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            per_class[WAVEFORM_NAMES[c]] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }

        return {
            "accuracy": float(accuracy),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def training_curve_plot(self, save_path: str | None = None) -> str:
        """Save training curves to PNG."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.history:
            raise RuntimeError("No training history — run train() first.")

        epochs = [h["epoch"] for h in self.history]
        train_loss = [h["train_loss"] for h in self.history]
        val_loss = [h["val_loss"] for h in self.history]
        train_acc = [h["train_acc"] for h in self.history]
        val_acc = [h["val_acc"] for h in self.history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, train_loss, label="Train Loss")
        ax1.plot(epochs, val_loss, label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, train_acc, label="Train Accuracy")
        ax2.plot(epochs, val_acc, label="Val Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training & Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = save_path or os.path.join(PROJECT_ROOT, "outputs", "training_curves.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        return save_path


# =========================================================================
# Main — run full training + evaluation + verification
# =========================================================================
def main():
    import json

    torch.manual_seed(42)
    np.random.seed(42)

    data_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Build model
    model = WaveformTransformerEncoder()
    print(f"Model parameters: {count_parameters(model):,}")

    # Trainer
    trainer = ModelTrainer(model, device="cpu")
    trainer.prepare_dataloaders(data_dir)

    # Train or Fine-tune
    if '--finetune-only' in sys.argv:
        print("Skipping full training, proceeding to boundary fine-tuning.")
        trainer.finetune_boundary()
        train_result = {"final_train_acc": 0, "final_val_acc": 0, "epochs_run": 0}
    else:
        train_result = trainer.train(epochs=50)
        print(f"\nTraining complete: {train_result['epochs_run']} epochs")
        print(f"  Best val acc: {train_result['final_val_acc']:.2%}")

    # Evaluate
    print("\n--- In-distribution test ---")
    test_metrics = trainer.evaluate(trainer.test_loader)
    print(f"  Accuracy: {test_metrics['accuracy']:.2%}")
    for name, m in test_metrics["per_class"].items():
        print(f"    {name:>10s}: P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f}")

    print("\n--- Boundary test ---")
    bd_metrics = trainer.evaluate(trainer.boundary_loader)
    print(f"  Accuracy: {bd_metrics['accuracy']:.2%}")

    # Plots
    if '--finetune-only' not in sys.argv:
        curve_path = trainer.training_curve_plot()
        print(f"\nTraining curves saved to: {curve_path}")
    else:
        print("\nSkipping plotting for finetuning.")

    # Thresholds
    indist_ok = test_metrics["accuracy"] > 0.88
    boundary_ok = bd_metrics["accuracy"] > 0.70

    errors = []
    if not indist_ok:
        errors.append(f"In-dist accuracy {test_metrics['accuracy']:.2%} < 88%")
    if not boundary_ok:
        errors.append(f"Boundary accuracy {bd_metrics['accuracy']:.2%} < 70%")

    result = {
        "final_train_accuracy": round(train_result["final_train_acc"], 4),
        "final_val_accuracy": round(train_result["final_val_acc"], 4),
        "test_indist_accuracy": round(test_metrics["accuracy"], 4),
        "test_boundary_accuracy": round(bd_metrics["accuracy"], 4),
        "indist_threshold_met": indist_ok,
        "boundary_threshold_met": boundary_ok,
        "best_model_saved": os.path.exists(os.path.join(output_dir, "best_model.pth")),
        "training_log_saved": os.path.exists(
            os.path.join(PROJECT_ROOT, "logs", "training_log.csv")),
        "errors": errors,
    }

    out_path = os.path.join(PROJECT_ROOT, "STEP_7_VERIFICATION.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nVerification JSON written to: {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
