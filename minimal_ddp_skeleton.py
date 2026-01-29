import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    # Initialize distributed communication
    # "gloo" works on CPU; use "nccl" for GPUs
    dist.init_process_group(backend="gloo")

    # Ensure identical initialization across processes
    torch.manual_seed(42)

def cleanup():
    # Tear down the distributed process group
    dist.destroy_process_group()

def main():
    setup()

    # Unique ID of this process
    rank = dist.get_rank()

    # Simple linear model
    model = torch.nn.Linear(10, 2)

    # Wrap model with DDP for gradient synchronization
    ddp_model = DDP(model)

    # Optimizer operates on DDP parameters
    optim = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

    # Classification loss
    loss_fn = torch.nn.CrossEntropyLoss()

    for step in range(20):
        # Fake batch (each rank gets different data)
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))

        optim.zero_grad()
        out = ddp_model(x)
        loss = loss_fn(out, y)

        # Backward triggers gradient all-reduce
        loss.backward()
        optim.step()

        # Only rank 0 logs
        if rank == 0 and step % 5 == 0:
            print("step", step, "loss", float(loss))

    cleanup()

if __name__ == "__main__":
    main()
