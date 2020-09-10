#!/bin/bash

if [[ $# -eq 1 ]]
then
    python -u run.py --run-type train --exp-config configs/$1.yaml --clear-only True
else
    echo "Expected args <variant> (ckpt)"
fi