#!/bin/bash

export PATH=$PATH:/home/cdsw/.local/bin

#ray head will be started with the same worker configurations resources
ray start --head --num-cpus=$worker_cpu --num-gpus=$worker_gpu --include-dashboard=true \
--dashboard-port=$CDSW_READONLY_PORT

cat /tmp/ray/ray_current_cluster > cluster_info.txt

echo ""
echo "https://read-only-$CDSW_ENGINE_ID.$CDSW_DOMAIN"