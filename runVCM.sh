#!/bin/bash

# usage introduction 
if [ $# -ne 1 ]; then
  echo "Usage: runVCM.sh <audiofile>"
  exit 1;
fi

# direction of scripts and set path 
export PATH=/home/${USER}/anaconda/bin:$PATH
SCRIPT=$(readlink -f $0)
BASEDIR=`dirname $SCRIPT`
cd $BASEDIR
echo $BASEDIR

# check results from Yunitator. If not, run Yunitator first to obtain yunitator_rttm_file
audio_file=$1 
bn=$(basename $audio_file)
dn=$(dirname $audio_file)
yunitator_rttm_file=$dn"/yunitator_"${bn//wav/rttm}  # yunicator output 
if [ ! -e $yunitator_rttm_file ]; then 
	echo "Error: Cannot find corresponding SAD outputs. Please run yunicatator first!"
	exit 1; 
fi 
vcm_rttm_file=$dn"/vcm_"${bn//wav/rttm} # vcm output 


# # make output folder for features, below input folder
# KEEPTEMP=false
# if [ $BASH_ARGV == "--keep-temp" ]; then
#     KEEPTEMP=true
# fi
# VCMTEMP=$dn/VCMtemp
# mkdir -p $VCMTEMP

# do vcm recognition 
python2 ./vcm_evaluate.py ${audio_file} ${yunitator_rttm_file} ${vcm_rttm_file}

# # simply remove segmented waves and acoustic features 
# if ! $KEEPTEMP; then
#     rm -rf $VCMTEMP
# fi


