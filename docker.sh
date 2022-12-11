cmd=/bin/bash
docker run --name "wholegraph_env" \
    --mount type=bind,source=/graph-learning,target=/graph-learning,readonly \
    --mount type=bind,source=$(pwd)/examples/gnn_v2,target=/app,readonly \
    --rm -it --gpus=all --ipc=host "wholegraph"  "$cmd"