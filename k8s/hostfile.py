import torch
import os
import socket

# Run this file with `torchrun --nproc_per_node=1 --nnodes=$WORLD_SIZE --node_rank=$RANK  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT hostfile.py > hostfile`

assert os.environ["LOCAL_RANK"] == "0", "This script should be run with torchrun with --nproc_per_node=1"

torch.distributed.init_process_group(backend="nccl")

rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()

hostname = socket.gethostname()
myline = f"{hostname} slots={torch.cuda.device_count()}"

all_hosts = [None] * world_size

torch.distributed.all_gather_object(all_hosts, myline)

print('\n'.join(all_hosts))

torch.distributed.destroy_process_group()
