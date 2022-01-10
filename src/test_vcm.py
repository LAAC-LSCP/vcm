#!usr/bin/env python
# -*- coding: utf8 -*-

import os
import textwrap

import pytest

from vcm import run_vcm
from utils import extract_feature, _clean

def test_feature_extraction(smilextract_bin_path):
    assert smilextract_bin_path is not None, "Missing OpenSMILE SMILExtract path! (--smilextract-bin-path option)"

    # Path to example file
    test_file_path = os.path.join(os.path.dirname(__file__), '../egs/example.wav')
    feature_output_path = test_file_path.replace('.wav', '.htk')
    _clean(feature_output_path)

    assert os.path.isfile(smilextract_bin_path), 'OpenSMILE SMILExtract not found!'.format(smilextract_bin_path)
    assert os.path.isfile(test_file_path), 'Example file not found!'

    feat_rc, feat_stdout, feat_stderr = extract_feature(test_file_path, feature_output_path, smilextract_bin_path)

    assert feat_rc == 0, 'OpenSMILE SMILExtract returned a non-zero return code! Standard Error: {}'.format(feat_stderr)
    assert os.path.isfile(feature_output_path), 'Output feature file not found!'
    _clean(feature_output_path)

def test_vcm_output(smilextract_bin_path):
    reference = textwrap.dedent("""\
        SPEAKER example 1 0.010 1.277 <NA> <NA> CRY 0.46 <NA>
        SPEAKER example 1 1.576 0.935 <NA> <NA> CRY 0.52 <NA>
        SPEAKER example 1 4.868 4.040 <NA> <NA> CRY 1.00 <NA>
        SPEAKER example 1 9.484 2.335 <NA> <NA> CRY 0.99 <NA>
        SPEAKER example 1 12.493 1.483 <NA> <NA> CRY 0.87 <NA>
        SPEAKER example 1 14.776 3.471 <NA> <NA> CRY 1.00 <NA>
        SPEAKER example 1 19.042 1.703 <NA> <NA> NCS 0.79 <NA>
        SPEAKER example 1 21.413 4.469 <NA> <NA> CRY 1.00 <NA>
        SPEAKER example 1 26.455 1.622 <NA> <NA> CRY 0.69 <NA>
        SPEAKER example 1 28.981 1.699 <NA> <NA> NCS 0.81 <NA>
        SPEAKER example 1 32.488 5.854 <NA> <NA> CRY 0.99 <NA>
        SPEAKER example 1 39.553 10.957 <NA> <NA> CRY 0.64 <NA>
        SPEAKER example 1 51.943 1.029 <NA> <NA> NCS 0.81 <NA>""")

    test_audio_path = os.path.join(os.path.dirname(__file__), '../egs/example.wav')
    test_rttm_path = os.path.join(os.path.dirname(__file__), '../egs/example.rttm')
    test_out_path = os.path.join(os.path.dirname(__file__), '../egs/example.vcm')
    run_vcm(smilextract_bin_path, test_audio_path, test_rttm_path, test_out_path,
            all_children=False, keep_other=False, reuse_temp=False, keep_temp=False)

    assert os.path.isfile(test_out_path), 'VCM output file {} is missing!'.format(test_out_path)

    with open(test_out_path) as out_predicted_file:
        predicted = ''.join(out_predicted_file.readlines())

    assert reference == predicted, "Reference output and predicted output are different!"
    os.remove(test_out_path)