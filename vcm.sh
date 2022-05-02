#!/bin/bash

usage() {
    # Lazy call to Python's help message
    python -um src.vcm --help | sed 's/vcm.py/vcm.sh/g' ;
    exit 1;
}

python -um src.vcm $*