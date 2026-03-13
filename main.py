import torch
import os
import pickle
from constants import VOCAB_SIZE, char2idx, PAD, DEVICE
from models import Seq2SeqTransformer, FlowNet
from env import AddSeqEnv
from train import train_baseline, train_adversarial
from test import test_robustness
from dataset import generate_addition_dataset, split_train_eval, build_metadata, save_dataset, verify_dataset_integrity
from utils import print_number_params

print(f"Vocab size: {VOCAB_SIZE}, Device: {DEVICE}")

# 1. Ensure Dataset Exists
train_file = "addition_dataset_train.pkl"
eval_file = "addition_dataset_eval.pkl"

if not os.path.exists(train_file) or not os.path.exists(eval_file):
    print("Generating dataset...")
    MAX_VAL = 99
    SEED = 42
    EVAL_RANGE = (40, 49)
    data = generate_addition_dataset(max_val=MAX_VAL, seed=SEED)
    train_data, eval_data = split_train_eval(data, eval_range=EVAL_RANGE)
    metadata = build_metadata(data, max_val=MAX_VAL, seed=SEED)
    save_dataset(train_file, train_data, metadata)
    save_dataset(eval_file, eval_data, metadata)
else:
    print("Dataset already generated. Loading...")

with open(train_file, "rb") as f:
    train_data = pickle.load(f)["data"]
with open(eval_file, "rb") as f:
    eval_data = pickle.load(f)["data"]

print(f"Loaded {len(train_data)} train samples, {len(eval_data)} eval samples.")

env = AddSeqEnv()

# 2. Baseline Training
baseline_model = Seq2SeqTransformer(VOCAB_SIZE, pad_idx=char2idx[PAD]).to(DEVICE)
print("\n--- Training Baseline Model ---")
print_number_params(baseline_model)
train_baseline(baseline_model, train_data, num_epochs=5)
print("\nTesting Baseline Model on Evaluation Set (OOD [40, 49]):")
baseline_acc = test_robustness(baseline_model, eval_data, num_samples=100)

# 3. Adversarial Training (LUCIDE)
# Clone baseline to start from the same point or start fresh.
# We will start from baseline as it's the correction phase.
llm_model = Seq2SeqTransformer(VOCAB_SIZE, pad_idx=char2idx[PAD]).to(DEVICE)
llm_model.load_state_dict(baseline_model.state_dict())
gflow_model = FlowNet().to(DEVICE)

print("\n--- Training Adversarial Model (LUCIDE Correction) ---")
print_number_params(gflow_model)
train_adversarial(llm_model, gflow_model, env, train_data, num_epochs=3, mix_ratio=0.5)

print("\nTesting Adversarial Model on Evaluation Set (OOD [40, 49]):")
adversarial_acc = test_robustness(llm_model, eval_data, num_samples=100)

if adversarial_acc > baseline_acc:
    print("\nSUCCESS: Adversarial LUCIDE is more robust!")
else:
    print("\nITERATE: Adversarial didn't improve accuracy.")
