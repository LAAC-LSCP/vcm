#!usr/bin/env python
# -*- coding: utf8 -*-

import os

import torch

torch.multiprocessing.set_start_method('spawn')

from ..src.vcm import run_vcm
from ..src.utils import extract_feature, read_text_file, _clean

TEST_AUDIO_PATH = os.path.join(os.path.dirname(__file__), '../egs/example.wav')
TEST_RTTM_PATH = os.path.join(os.path.dirname(__file__), '../egs/example.rttm')
TEST_OUT_PATH = os.path.join(os.path.dirname(__file__), '../egs/example.vcm')
REFERENCE_PATH = os.path.join(os.path.dirname(__file__), 'references/example_all-children-{}_remove-others-{}.txt')


def test_feature_extraction(smilextract_bin_path):
    assert smilextract_bin_path is not None, "Missing OpenSMILE SMILExtract path! (--smilextract-bin-path option)"

    # Path to example file
    test_file_path = os.path.join(os.path.dirname(__file__), '../egs/example.wav')
    feature_output_path = test_file_path.replace('.wav', '.htk')
    _clean(feature_output_path)

    assert os.path.isfile(smilextract_bin_path), 'OpenSMILE SMILExtract not found!'.format(smilextract_bin_path)
    assert os.path.isfile(test_file_path), 'Example file not found!'

    extract_feature(test_file_path, feature_output_path, smilextract_bin_path)

    assert os.path.isfile(feature_output_path), 'Output feature file not found!'
    _clean(feature_output_path)


def test_vcm_outputs(smilextract_bin_path):
    for all_children in [True, False]:
        for remove_others in [True, False]:
            run_vcm(smilextract_bin_path, TEST_AUDIO_PATH, TEST_RTTM_PATH, TEST_OUT_PATH, keep_temp=False, n_jobs=1,
                    all_children=all_children, remove_others=remove_others, reuse_temp=False, skip_done=False)

            assert os.path.isfile(TEST_OUT_PATH), \
                'VCM output file {} is missing! ' \
                'all_children = {}, remove_others = {}'.format(TEST_OUT_PATH, all_children, remove_others)

            reference = '\n'.join(read_text_file(REFERENCE_PATH.format(all_children, remove_others)))
            predicted = '\n'.join(read_text_file(TEST_OUT_PATH))

            assert reference == predicted, \
                "Reference output and predicted output are different! " \
                "all_children = {}, remove_others = {}".format(TEST_OUT_PATH, all_children, remove_others)
    os.remove(TEST_OUT_PATH)
