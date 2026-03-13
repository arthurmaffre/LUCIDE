import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from constants import DEVICE, char2idx, idx2char, BOS, EOS, PAD, VOCAB_SIZE
from models import Seq2SeqTransformer, FlowNet
from env import AddSeqEnv
from utils import sample_trajectory, tb_loss, compute_bayesian_divergence, parse_and_compute_target

def train_baseline(llm_model: Seq2SeqTransformer, train_data, num_epochs=100, batch_size=128):
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx[PAD])
    optimizer = optim.Adam(llm_model.parameters(), lr=5e-4)
    for epoch in range(num_epochs):
        llm_model.train()
        random.shuffle(train_data)
        total_loss = 0.0
        num_batches = len(train_data) // batch_size
        if num_batches == 0:
            num_batches = 1
        for i in range(num_batches):
            batch = train_data[i*batch_size : (i+1)*batch_size]
            srcs, tgts = [], []
            for input_str, target_str in batch:
                src = torch.tensor([char2idx[BOS]] + [char2idx.get(c, char2idx[PAD]) for c in input_str], device=DEVICE)
                tgt = torch.tensor([char2idx[BOS]] + [char2idx.get(c, char2idx[PAD]) for c in target_str] + [char2idx[EOS]], device=DEVICE)
                srcs.append(src)
                tgts.append(tgt)

            src_padded = pad_sequence(srcs, batch_first=True, padding_value=char2idx[PAD])
            tgt_padded = pad_sequence(tgts, batch_first=True, padding_value=char2idx[PAD])

            tgt_input = tgt_padded[:, :-1]
            tgt_output = tgt_padded[:, 1:].contiguous().view(-1)

            logits = llm_model(src_padded, tgt_input).contiguous().view(-1, VOCAB_SIZE)
            loss = criterion(logits, tgt_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        if (epoch+1) % 5 == 0:
            print(f"Baseline Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        if avg_loss < 0.1: # early stop if well-trained
            print(f"Early stopping at epoch {epoch+1}")
            break
    return llm_model

def train_adversarial(llm_model: Seq2SeqTransformer, gflow_model: FlowNet, env, train_data, num_epochs=20, batch_size=128, mix_ratio=0.5):
    llm_optimizer = optim.Adam(llm_model.parameters(), lr=1e-4) # Lower LR for fine-tuning
    gflow_optimizer = optim.Adam(gflow_model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx[PAD])

    for epoch in range(num_epochs):
        # Phase 1: GFlowNet maximizes divergence
        gflow_model.train()
        gflow_losses, divergences = [], []
        trajs, rewards = [], []
        for _ in range(batch_size):
            traj, reward = sample_trajectory(gflow_model, env)
            prefix = torch.tensor([traj[:-1]], device=DEVICE)
            token = torch.tensor([traj[-1]], device=DEVICE)
            prior_prob = 1.0 / len(traj)

            # Use dummy action_prob/info_prob for the original formula
            action_prob = random.uniform(0.5, 1.0)
            info_prob = random.uniform(0.5, 1.0)

            div = compute_bayesian_divergence(llm_model, prefix, token, prior_prob, gflow_model=gflow_model)
            reward += div.item()
            trajs.append(traj)
            rewards.append(reward)
            divergences.append(div.item())

        gflow_loss = tb_loss(gflow_model, trajs, rewards)
        gflow_optimizer.zero_grad()
        gflow_loss.backward()
        gflow_optimizer.step()
        gflow_losses.append(gflow_loss.item())

        # Phase 2: LLM minimizes on mixed batch (adversarial + real)
        llm_model.train()
        llm_losses = []
        num_generated = int(batch_size * mix_ratio)
        num_real = batch_size - num_generated

        srcs, tgts = [], []

        # Generated (Compute targets, essentially treating adversary as OOD generator)
        for _ in range(num_generated):
            traj, _ = sample_trajectory(gflow_model, env)
            seq_str = ''.join(idx2char[t] for t in traj[1:-1])
            target_str = parse_and_compute_target(seq_str + '=') # Hack to parse AddSeqEnv correctly
            if target_str:
                src = torch.tensor([char2idx[BOS]] + [char2idx.get(c, char2idx[PAD]) for c in seq_str + '='], device=DEVICE)
                tgt = torch.tensor([char2idx[BOS]] + [char2idx.get(c, char2idx[PAD]) for c in target_str] + [char2idx[EOS]], device=DEVICE)
                srcs.append(src)
                tgts.append(tgt)

        # Real
        real_samples = random.sample(train_data, min(num_real, len(train_data)))
        for input_str, target_str in real_samples:
            src = torch.tensor([char2idx[BOS]] + [char2idx.get(c, char2idx[PAD]) for c in input_str], device=DEVICE)
            tgt = torch.tensor([char2idx[BOS]] + [char2idx.get(c, char2idx[PAD]) for c in target_str] + [char2idx[EOS]], device=DEVICE)
            srcs.append(src)
            tgts.append(tgt)

        if len(srcs) > 0:
            src_padded = pad_sequence(srcs, batch_first=True, padding_value=char2idx[PAD])
            tgt_padded = pad_sequence(tgts, batch_first=True, padding_value=char2idx[PAD])

            tgt_input = tgt_padded[:, :-1]
            tgt_output = tgt_padded[:, 1:].contiguous().view(-1)

            logits = llm_model(src_padded, tgt_input).contiguous().view(-1, VOCAB_SIZE)
            ce_loss = criterion(logits, tgt_output)

            loss = ce_loss
            llm_optimizer.zero_grad()
            loss.backward()
            llm_optimizer.step()
            llm_losses.append(loss.item())

        print(f"Adversarial Epoch {epoch+1}: GFlow Loss={np.mean(gflow_losses):.4f}, Avg Div={np.mean(divergences):.4f}, LLM Loss={np.mean(llm_losses):.4f}")
    return llm_model
