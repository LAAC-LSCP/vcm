#!/bin/bash

usage() {
    # Lazy call to Python's help message
    python -um src.vcm --help | sed 's/vcm.py/vcm.sh/g' ;
    exit 1;
}

if [[ $# -eq 0 ]]; then
  usage;
  exit 1;
fi

python -um src.vcm $*