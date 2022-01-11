#!/bin/bash

usage() {
    # Lazy call to Python's help message
    python $RUN_PATH --help | sed 's/vcm.py/vcm.sh/g'
    exit 1;
}

RUN_PATH='./src/vcm.py'

INPUT_AUDIO_PATH=''
INPUT_RTTM_PATH=''
SMILEXTRACT_BIN_PATH=''
OUTPUT_VCM_PATH=''
AUDIO_EXTENSION=''
ALL_CHILDREN=''
KEEP_OTHER=''
KEEP_TEMP=''
N_JOBS=''
REUSE_TEMP=''
SKIP_DONE=''

# Arguments parser
while [[ $# -gt 0 ]]; do
  case $1 in
    -a|--input-audio-path)
    INPUT_AUDIO_PATH=$2
    shift
    shift
    ;;
    -r|--input-rttm-path)
    INPUT_RTTM_PATH=$2
    shift
    shift
    ;;
    -s|--smilextract-bin-path)
    SMILEXTRACT_BIN_PATH=$2
    shift
    shift
    ;;
    -o|--output-vcm-path)
    OUTPUT_VCM_PATH='--output-vcm-path '$2
    shift
    shift
    ;;
    -x|--audio-extension)
    AUDIO_EXTENSION='--audio-extension '$2
    shift
    shift
    ;;
    -j|-J|--n-jobs)
    N_JOBS='--n-jobs '$2
    shift
    shift
    ;;
    --all-children)
    ALL_CHILDREN='--all-children'
    shift # past argument
    ;;
    --keep-other)
    KEEP_OTHER='--keep-other'
    shift # past argument
    ;;
    --keep-temp)
    KEEP_TEMP='--keep-temp'
    shift # past argument
    ;;
    --reuse-temp)
    REUSE_TEMP='--reuse-temp'
    shift # past argument
    ;;
    --skip-done)
    SKIP_DONE='--skip-done'
    shift # past argument
    ;;
    -h|--help)
    shift
    usage
    ;;
    *)
    echo "/!\ Unknown option '$1'. Not running anything" # Unknown option
    usage
    ;;
  esac
done

if [[ -z $INPUT_AUDIO_PATH ]] | [[ -z $INPUT_RTTM_PATH ]] | [[ -z $SMILEXTRACT_BIN_PATH ]]; then
  usage
fi

python $RUN_PATH --input-audio-path $INPUT_AUDIO_PATH --input-rttm-path $INPUT_RTTM_PATH --smilextract-bin-path $SMILEXTRACT_BIN_PATH \
                 $OUTPUT_VCM_PATH $AUDIO_EXTENSION $ALL_CHILDREN $KEEP_OTHER $KEEP_TEMP $REUSE_TEMP $N_JOBS $SKIP_DONE
