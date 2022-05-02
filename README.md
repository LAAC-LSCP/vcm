# VCM

The original code to this VCM is to be found [here](https://github.com/srvk/vcm). 

The code was cleaned and tested by William N. Havard (07&10/01/22).

## Installation

Clone this repository.
for example if you use ssh:
```
git clone git@github.com:LAAC-LSCP/vcm.git
```


If you which to install the dependencies directly you can run:
```
pip install -r requirements.txt
```
If you want to keep everything in a conda environment:
```
conda create -p /my/environment/vcm pip
conda activate /my/environment/vcm
pip install -r requirements
```
You will need the SMILExtract binary file to run vcm, download it for example in you vcm directory:
```
wget https://github.com/georgepar/opensmile/blob/master/bin/linux_x64_standalone_static/SMILExtract?raw=true -O SMILExtract
chmod u+x SMILExtract
```

## Run

If you use conda, remember to check that your environment is activated
```
conda activate /my/environment/vcm
(vcm) myname@mycomputer$
```
Vcm is used with the vcm.sh script:
```
usage: vcm.py [-h] -a INPUT_AUDIO_PATH -r INPUT_RTTM_PATH -s
              SMILEXTRACT_BIN_PATH [-o OUTPUT_VCM_PATH]
              [-x AUDIO_EXTENSION] [--all-children] [--remove-others]
              [--from-batched-vtc] [--keep-temp] [--reuse-temp]
              [--temp-dir TEMP_DIR] [--skip-done] [-j N_JOBS]

optional arguments:
  -h, --help            show this help message and exit
  
  -a INPUT_AUDIO_PATH, --input-audio-path INPUT_AUDIO_PATH
                        Path to the audio file to be processed.
  -r INPUT_RTTM_PATH, --input-rttm-path INPUT_RTTM_PATH
                        Path to the VTC output of the file to be processed.
  -s SMILEXTRACT_BIN_PATH, --smilextract-bin-path SMILEXTRACT_BIN_PATH
                        Path to smilextract SMILExtract (v2.3) binary.
  -o OUTPUT_VCM_PATH, --output-vcm-path OUTPUT_VCM_PATH
                        Output path were the results of the VCM should be
                        stored. Default: Same as RTTM file.
  -x AUDIO_EXTENSION, --audio-extension AUDIO_EXTENSION
                        Audio files file extension (no extension '' also
                        accepted). Default: '.wav'
                        
  --all-children        Should speech segment produced by other children
                        than the key child (KCHI) be analysed. (Default:
                        False.)
  --remove-others       Should the VTC annotations for the other speakers
                        be removed from the VCMoutput file. If Segments
                        from speaker-type SPEECH, MAL, FEM, etc. will be
                        removed. (Default: False.)
  --from-batched-vtc    Whether the VTC files were generated using
                        LSCP/LAAC batch-voice-type-classifier or not./!\
                        LSCP/LAAC specific, you shouldn't be needing this
                        option. (Default: False.)
                        
  --keep-temp           Whether temporary file should be kept or not.
                        (Default: False.)
  --reuse-temp          Whether temporary file should be reused instead of
                        being recomputed. (Default: False.)
  --temp-dir TEMP_DIR   Set path to temporary directory. (Default:
                        `../tmp`).
  --skip-done           Whether RTTM for which a VCM file already exists
                        should be skipped. (Default: False.)
                        
  -j N_JOBS, -J N_JOBS, --n-jobs N_JOBS
                        Number of parallel jobs to run.
```
Launch you computation:
```
./vcm.sh -a audio/file/or/directory -r rttm/file/or/directory -s path/SMILExtract -o output/path -j 8
```

## Tests

To test the installation, run the following command:

```bash
pytest tst --smilextract-bin-path=/scratch2/whavard/PACKAGES/opensmile/bin/linux_x64_standalone_static/SMILExtract
```

The original code was written in Python 2 and used an unknown version of Pytorch (presumably 0.3.0). The code however
runs seemlessly with Python 3.7 and Pytorch 1.6. `requirements.txt` is only given for reproducibility purposes, and
packages with lower version numbers might work as well.

## Notes

* example.rttm was extracted using Marvin's VTC as packaged by ALICE
* Feature extraction is done using OpenSMILE v2.3. Code was run and tested with the precompile version: [opensmile/bin)/**linux_x64_standalone_static**/](https://github.com/georgepar/opensmile/tree/master/bin/linux_x64_standalone_static)
* *LSCP/LAAC specific*: To process RTTM files generated with [the batched version of the voice-type-classificier](https://github.com/lucasgautheron/batch-voice-type-classifier) use the following option `--from-batched-vtc`. 

# References

* Extracted features (88 eGeMAPS) reference: "[The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing](https://sail.usc.edu/publications/files/eyben-preprinttaffc-2015.pdf)". Eyben et al., 2015.
* VCM
  reference: "[VCMNet: Weakly Supervised Learning for Automatic Infant Vocalisation Maturity Analysis](https://dl.acm.org/doi/10.1145/3340555.3353751)"
  . Futaisi et al., 2019. [DOI: 10.1145/3340555.3353751](https://doi.org/10.1145/3340555.3353751)
* [VTC code](https://github.com/MarvinLvn/voice-type-classifier/tree/new_model#66f87c2a8cef25c80c9d9b91f4023ab4757413da) (
  Lavechin et al.)
* [ALICE Code](https://github.com/orasanen/ALICE) (Räsänen et al.)
