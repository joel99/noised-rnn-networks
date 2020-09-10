#!/bin/bash

if [[ $# -eq 2 ]]
then
    python -u run.py --run-type eval --exp-config configs/$1.yaml --ckpt-path $1.$2.pth
else
    echo "Expected args <variant> (ckpt)"
fi