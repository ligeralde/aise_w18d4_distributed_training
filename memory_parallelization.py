import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
from torch.utils.data import DataLoader, Dataset, DistributedSampler


# -------------------------
# Simple config
# -------------------------
@dataclass
class DemoConfig:
    # Use a moderately larger model so that
    # parameter counts and communication costs
    # are more visible in the metrics.
    input_dim: int = 256
    hidden_dim: int = 1024
    output_dim: int = 10
    dataset_size: int = 2048
    batch_size: int = 64
    epochs: int = 3

# -------------------------
# Dataset
# -------------------------
class SmallDataset(Dataset):
    def __init__(self, size=128, input_dim=10, output_dim=1):
        self.X = torch.randn(size, input_dim)
        self.y = torch.randn(size, output_dim)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------
# Tiny models
# -------------------------
class TinyModel(nn.Module):
    """Baseline tiny MLP; used for DDP demos."""

    def __init__(self, input_dim=10, hidden_dim=32, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class TPLinear(nn.Module):
    """
    Column-sharded Linear layer across ranks.

    - Each rank holds a slice of the output features.
    - Forward: local matmul then all_gather to form full output.
    - Backward: handled by autograd, but communication cost is visible
      via the timed all_gather.
    """

    def __init__(self, in_features: int, out_features: int, pg, rank: int, world_size: int):
        super().__init__()
        assert (
            out_features % world_size == 0
        ), "For this demo, out_features must be divisible by world_size"

        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.pg = pg

        self.local_out_features = out_features // world_size
        self.linear = nn.Linear(in_features, self.local_out_features)

        # Metrics
        self.reset_comm_stats()

    def reset_comm_stats(self):
        self.comm_calls = 0
        self.comm_time = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_out = self.linear(x)

        # Gather outputs from all ranks
        gather_list = [torch.zeros_like(local_out) for _ in range(self.world_size)]

        t0 = time.time()
        torch.distributed.all_gather(gather_list, local_out, group=self.pg)
        t1 = time.time()

        self.comm_calls += 1
        self.comm_time += t1 - t0

        # Concatenate along the feature dimension to form full output
        full_out = torch.cat(gather_list, dim=-1)
        return full_out


class TinyTPModel(nn.Module):
    """
    Tiny model that uses TPLinear for its hidden layer.
    """

    def __init__(self, cfg: DemoConfig, pg, rank: int, world_size: int):
        super().__init__()
        self.fc1 = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.relu1 = nn.ReLU()
        self.tp_hidden = TPLinear(cfg.hidden_dim, cfg.hidden_dim, pg=pg, rank=rank, world_size=world_size)
        self.relu2 = nn.ReLU()
        self.fc_out = nn.Linear(cfg.hidden_dim, cfg.output_dim)

        self._tp_module = self.tp_hidden

    @property
    def tp_module(self) -> TPLinear:
        return self._tp_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.tp_hidden(x))
        x = self.fc_out(x)
        return x

# -------------------------
# Training
# -------------------------
def count_parameters(m: nn.Module) -> int:
    """Count total parameters in a module."""
    return sum(p.numel() for p in m.parameters())


def count_tp_model_local_params(model: nn.Module) -> int:
    """
    Count only locally-held parameters in a TP model.
    
    For TPLinear layers, counts only the local Linear layer parameters,
    not the full sharded layer size.
    """
    total = 0
    for name, param in model.named_parameters():
        # All parameters are already local (TPLinear only stores local Linear layer)
        total += param.numel()
    return total


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    rank: int,
    mode: str,
    global_param_count: int,
    local_param_count: int,
    tp_comm_stats: Optional[Dict[str, float]] = None,
    ddp_comm_stats: Optional[Dict[str, float]] = None,
) -> float:
    model.train()
    total_loss = 0.0
    batch_counts = []
    start = time.time()
    backward_times = []
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = nn.MSELoss()(out, y)
        
        # Measure backward pass time (includes DDP gradient sync)
        backward_start = time.time()
        loss.backward()
        backward_end = time.time()
        backward_times.append(backward_end - backward_start)
        
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        batch_counts.append(X.size(0))
    end = time.time()
    
    # Calculate DDP communication stats
    if ddp_comm_stats is not None and backward_times:
        ddp_comm_stats['calls'] = len(backward_times)
        ddp_comm_stats['total_time'] = sum(backward_times)
        ddp_comm_stats['avg_time_ms'] = (sum(backward_times) / len(backward_times)) * 1000.0

    # Metrics
    throughput = len(dataloader.dataset) / (end - start)
    gpu_mem = torch.cuda.memory_allocated(device) / 1024 ** 2 if torch.cuda.is_available() else 0
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5

    comm_str = ""
    # Communication stats are logged separately after each epoch, not in per-epoch line

    print(
        f"[Mode {mode.upper()} | Rank {rank}] "
        f"Loss: {total_loss/len(dataloader.dataset):.4f} | "
        f"Throughput: {throughput:.2f} samples/sec | "
        f"Grad norm: {grad_norm:.4f} | "
        f"Global params: {global_param_count} | "
        f"Local params: {local_param_count} | "
        f"GPU Mem (MB): {gpu_mem:.1f} | "
        f"Batches: {batch_counts}"
        f"{comm_str}"
    )

    return total_loss / len(dataloader.dataset)

# -------------------------
# DDP setup (CPU)
# -------------------------
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    torch.distributed.destroy_process_group()

def main_worker(rank: int, world_size: int, mode: str, cfg: DemoConfig):
    setup(rank, world_size)

    # Seed everything for deterministic behavior
    seed = 42
    torch.manual_seed(seed + rank)
    import random
    import numpy as np

    random.seed(seed + rank)
    np.random.seed(seed + rank)

    device = torch.device("cpu")

    dataset = SmallDataset(size=cfg.dataset_size, input_dim=cfg.input_dim, output_dim=cfg.output_dim)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler)

    # Build base model for global param counting
    # Use the same base architecture for both modes for fair comparison
    base_model_dense = TinyModel(cfg.input_dim, cfg.hidden_dim, cfg.output_dim)
    global_param_count = count_parameters(base_model_dense)  # Full model size
    
    tp_module = None
    ddp_comm_stats = None
    if mode == "ddp":
        base_model = base_model_dense.to(device)
        model = DDP(base_model)
        wrap_type = "DDP"
        # DDP replicates full model on each rank
        local_param_count = count_parameters(model)
        # Track DDP gradient synchronization
        ddp_comm_stats = {'calls': 0, 'total_time': 0.0, 'avg_time_ms': 0.0}
    elif mode == "tp":
        # For TP, replace the middle Linear layer with TPLinear
        base_model = TinyTPModel(cfg, pg=torch.distributed.group.WORLD, rank=rank, world_size=world_size).to(device)
        model = base_model
        wrap_type = "TP"
        tp_module = model.tp_module
        # TP shards parameters - count only local shards
        local_param_count = count_tp_model_local_params(model)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if rank == 0:
        print(
            f"\n=== Starting {wrap_type} demo ===\n"
            f"World size: {world_size} | "
            f"Global params (rank 0 view): {global_param_count} | "
            f"Local params (rank 0): {local_param_count}"
        )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train
    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        if rank == 0:
            print(f"\n=== Epoch {epoch} ({wrap_type}) ===")

        # Reset communication stats each epoch
        if tp_module is not None:
            tp_module.reset_comm_stats()
        if ddp_comm_stats is not None:
            ddp_comm_stats['calls'] = 0
            ddp_comm_stats['total_time'] = 0.0
            ddp_comm_stats['avg_time_ms'] = 0.0

        loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            device,
            rank,
            mode,
            global_param_count,
            local_param_count,
            tp_comm_stats=None,
            ddp_comm_stats=ddp_comm_stats,
        )

        # After epoch, log communication stats (rank 0 only)
        if tp_module is not None and rank == 0:
            steps = max(tp_module.comm_calls, 1)
            avg_ms = (tp_module.comm_time / steps) * 1000.0
            print(
                f"[Mode {mode.upper()} | Rank 0] "
                f"TP all_gather calls this epoch: {int(tp_module.comm_calls)} | "
                f"Avg TP all_gather time/step: {avg_ms:.3f} ms"
            )
        elif ddp_comm_stats is not None and rank == 0:
            print(
                f"[Mode {mode.upper()} | Rank 0] "
                f"DDP grad sync calls this epoch: {int(ddp_comm_stats['calls'])} | "
                f"Avg DDP grad sync time/step: {ddp_comm_stats['avg_time_ms']:.3f} ms"
            )

    # Checkpoint only by rank 0
    if rank == 0:
        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
        }
        torch.save(ckpt, f"checkpoint_{wrap_type}.pth")
        print(f"Checkpoint saved by rank 0 ({wrap_type})")

    cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal DDP / TP demo on CPU.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ddp", "tp"],
        required=True,
        help="Distributed strategy to run: ddp or tp.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Number of processes / ranks to launch.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size per rank.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=2048,
        help="Total synthetic dataset size.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Start from DemoConfig defaults (larger model), override size-related
    # fields from CLI so we can quickly scale the run up or down.
    cfg = DemoConfig(
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    print(
        f"Launching {args.world_size} processes for CPU {args.mode.upper()} demo "
        f"(epochs={args.epochs}, batch_size={args.batch_size}, dataset_size={args.dataset_size})..."
    )
    spawn(
        main_worker,
        args=(args.world_size, args.mode, cfg),
        nprocs=args.world_size,
    )
