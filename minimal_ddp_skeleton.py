import os 
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    dist.init_process_group(backend="gloo")  # "nccl" for GPU
    torch.manual_seed(42)

def cleanup():
    dist.destroy_process_group()

def main():
    setup()

    rank = dist.get_rank()
    model = torch.nn.Linear(10, 2)
    ddp_model = DDP(model)

    optim = torch.optim.SGD(ddp_model.parameters(), lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    for step in range(20):
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        optim.zero_grad()
        out = ddp_model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optim.step()
        if rank == 0 and step % 5 == 0:
            print("step", step, "loss", float(loss))

    cleanup()

if __name__ == "__main__":
    main()
