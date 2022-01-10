# VCM

The original code to this VCM is to be found [here](https://github.com/srvk/vcm). 

The code was cleaned and tested by William N. Havard (07&10/01/22).

## Tests

To test the installation, run the following command:

```bash
pytest ./src/ --smilextract-bin-path=/scratch2/whavard/PACKAGES/opensmile/bin/linux_x64_standalone_static/SMILExtract
```

The original code was written in Python 2 and used an unknow version of Pytorch (presumably 0.3.0). The code however runs seemlessly with Python 3.7 and Pytorch 1.6. `requirements.txt` is only given for reproducibility purposes, and packages with lower version numbers might work as well.

## Notes

* example.rttm was extracted using Marvin's VTC as packaged by ALICE
* Feature extraction is done using OpenSMILE v2.3. Code was run and tested with the precompile version: [opensmile/bin)/**linux_x64_standalone_static**/](https://github.com/georgepar/opensmile/tree/master/bin/linux_x64_standalone_static)

# References

* Extracted features (88 eGeMAPS) reference: "[The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing](https://sail.usc.edu/publications/files/eyben-preprinttaffc-2015.pdf)". Eyben et al., 2015.
* VCM reference: "[VCMNet: Weakly Supervised Learning for Automatic Infant Vocalisation Maturity Analysis](https://dl.acm.org/doi/10.1145/3340555.3353751)". Futaisi et al., 2019. [DOI: 10.1145/3340555.3353751](https://doi.org/10.1145/3340555.3353751)
* [VTC code](https://github.com/MarvinLvn/voice-type-classifier) (Lavechin et al.)
* [ALICE Code](https://github.com/orasanen/ALICE) (Räsänen et al.)
