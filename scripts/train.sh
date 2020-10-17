#!/bin/bash

if [[ $# -eq 1 ]]
then
    echo "Here we go"
    python -u run.py --run-type train --exp-config configs/$1.yaml
elif [[ $# -ge 2 ]]
# Simply add the "-s" flag in order to sweep a directory (will run sequentially)
then
    echo ${@:3}
    python -u run.py --run-type train --exp-config configs/$1.yaml ${@:2}
# elif [[ $# -eq 3 ]]
# then
#     python -u run.py --run-type train --exp-config configs/$1.yaml --ckpt-path $1.$3.pth MODEL.GRAPH_FILE data/configs/graphs/$2
else
    echo "Expected args <variant> (ckpt)"
fi