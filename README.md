# Basic setup for showcasing ray and vllm on CML


## setup.
0. add the following env vars to your project
worker_cpu=8
worker_gpu=1
num_workers=1
worker_ram_memory=16
1. go to site administration and add a new profile with **8 cpus** and **16GB RAM** 
2. add the following cml runtime. (If you would like to use your own image. Refer to Runtime section for further steps [tba])
**luismap/cml:pbjcuda-V2.0**
3. install requirements
```bash
pip install -r requirements.txt
```
4. create ray cluster.
This script will create a ray cluster with 2 nodes.
```bash
python3 ray_start_cluster_python.py
```
5. run the script mistral_vllm.py.