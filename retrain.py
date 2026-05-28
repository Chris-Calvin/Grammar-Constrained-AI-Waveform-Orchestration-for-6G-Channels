"""Retrain with tuned hyperparameters to push past 88% in-dist accuracy."""
import sys
import os
import json
import torch
import numpy as np

# Dynamically set project root based on this script's directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.transformer.model import WaveformTransformerEncoder, count_parameters
from src.transformer.trainer import ModelTrainer

torch.manual_seed(42)
np.random.seed(42)

model = WaveformTransformerEncoder()
print(f"Parameters: {count_parameters(model):,}")

trainer = ModelTrainer(model, config={
    "epochs": 80,
    "lr": 2e-3,
    "patience": 12,
    "weight_decay": 1e-5,
    "batch_size": 128,
}, device="cpu")

processed_data_dir = os.path.join(PROJECT_ROOT, "data", "processed")
trainer.prepare_dataloaders(processed_data_dir)
result = trainer.train(epochs=80)
print(f"Val acc: {result['final_val_acc']:.4f}")

test_m = trainer.evaluate(trainer.test_loader)
bd_m = trainer.evaluate(trainer.boundary_loader)
print(f"Test indist: {test_m['accuracy']:.4f}")
print(f"Test boundary: {bd_m['accuracy']:.4f}")

trainer.training_curve_plot()

errors = []
indist_ok = test_m["accuracy"] > 0.88
boundary_ok = bd_m["accuracy"] > 0.70
if not indist_ok:
    errors.append(f"In-dist accuracy {test_m['accuracy']:.2%} < 88%")
if not boundary_ok:
    errors.append(f"Boundary accuracy {bd_m['accuracy']:.2%} < 70%")

v = {
    "final_train_accuracy": round(result["final_train_acc"], 4),
    "final_val_accuracy": round(result["final_val_acc"], 4),
    "test_indist_accuracy": round(test_m["accuracy"], 4),
    "test_boundary_accuracy": round(bd_m["accuracy"], 4),
    "indist_threshold_met": indist_ok,
    "boundary_threshold_met": boundary_ok,
    "best_model_saved": os.path.exists(os.path.join(PROJECT_ROOT, "outputs", "best_model.pth")),
    "training_log_saved": os.path.exists(os.path.join(PROJECT_ROOT, "logs", "training_log.csv")),
    "errors": errors,
}

verification_file = os.path.join(PROJECT_ROOT, "STEP_7_VERIFICATION.json")
with open(verification_file, "w") as f:
    json.dump(v, f, indent=2)
print(json.dumps(v, indent=2))

