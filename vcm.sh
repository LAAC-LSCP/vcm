#!/bin/bash

usage() {
    # Lazy call to Python's help message
    python $RUN_PATH --help | sed 's/vcm.py/vcm.sh/g' ;
    exit 1;
}

RUN_PATH='./src/vcm.py'
python $RUN_PATH $*