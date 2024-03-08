import subprocess
import cml.workers_v1 as cdsw
import os


subprocess.call("ray_start_head.sh", shell=True)

with open("cluster_info.txt", "r") as file:
    ray_head_addr = file.readline()

worker_start_cmd = f"!export PATH=$PATH:/home/cdsw/.local/bin; ray start --block --address={ray_head_addr}"


num_workers = int(os.environ["num_workers"])  # defaults to 1
memory = int(os.environ["worker_ram_memory"])  # default to 16
cpu = int(os.environ["worker_cpu"])  # defaults to 8
gpu = int(
    os.environ["worker_gpu"]
)  # defaults to 1.  # default to 1. tensors needs a multiple of the attentions layers
timeout_seconds = 900  # timeout for workers api to wait for resources assingment

ray_workers = cdsw.launch_workers(
    n=num_workers, cpu=cpu, memory=memory, nvidia_gpu=gpu, code=worker_start_cmd
)

ray_worker_details = cdsw.await_workers(
    ray_workers, wait_for_completion=False, timeout_seconds=timeout_seconds
)

ray_worker_details
