#!usr/bin/env python
# -*- coding: utf8 -*-

import os
from subprocess import PIPE, run

import librosa
import soundfile
import tqdm
from tqdm import tqdm

RTTM_SEP = ' '
RTTM_LINE_PATTERN = "SPEAKER {} 1 {} {} <NA> <NA> {} {:.2f} <NA>".replace(' ',
                                                                          RTTM_SEP)  # fn, onset, duration,
# vcm-class, conf


#
# General utility functions
#

def _clean(feature_output_path):
    if os.path.isfile(feature_output_path):
        os.remove(feature_output_path)


def get_raw_filename(path):
    return os.path.splitext(os.path.basename(os.path.normpath(path)))[0]


def get_path_suffix(path_a, path_b):
    path_a = os.path.split(path_a)[0]
    return path_a.removeprefix(os.path.commonpath([path_a, path_b])).removeprefix('/')


def read_text_file(input_path):
    with open(input_path) as file_in:
        return [line.strip() for line in file_in]


def dump_text_file(output_path, lines):
    with open(output_path, 'w') as file_out:
        file_out.write('\n'.join(lines))


def find_all_files(path, extension=''):
    audiofile_name2path = {}

    for p, d, f in tqdm(os.walk(path), leave=False, desc='Scanning for `{}` files...'.format(extension)):
        for file in f:
            if extension != '':
                if file.endswith(extension):
                    audiofile_name2path[get_raw_filename(file)] = os.path.join(p, file)
            else:
                audiofile_name2path[get_raw_filename(file)] = os.path.join(p, file)
    return audiofile_name2path

def _write_log(errors, path):
    log_fn = 'log_{}.log'.format(os.path.dirname(os.path.normpath(path)).strip(os.sep).replace(os.sep, '-'))
    with open(log_fn, 'a+') as out_log_file:
        out_log_file.write('\n'.join(errors))
    return log_fn

#
# VCM utility functions
#

def extract_feature(audio_input_path, feature_output_path, SMILEXTRACT_PATH):
    config = os.path.normpath(os.path.join(os.path.dirname(__file__), '../config/gemaps/eGeMAPSv01a.conf'))
    cmd = f"{SMILEXTRACT_PATH} " \
          f"-C {config} " \
          f"-I {audio_input_path} " \
          f"-htkoutput {feature_output_path} " \
          f"-nologfile 1".split(' ')
    result = run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    r_rc, r_stdout, r_stderr = result.returncode, result.stdout, result.stderr

    assert result.returncode == 0, 'OpenSMILE SMILExtract returned a non-zero ({}) ' \
                                   'exit code for file {}!\n{}'.format(r_rc, audio_input_path, r_stderr)
    assert os.path.isfile(feature_output_path), "Error: Feature file {} for file {} was " \
                                                "not generated properly!".format(feature_output_path, audio_input_path)
    return

def seg_audio(input_audio, output_audio, onset, duration):
    # onset and duration should be values provided in SECONDS
    try:
        wave_data, sr = librosa.load(input_audio, sr=None, offset=float(onset), duration=float(duration))
        soundfile.write(output_audio, wave_data, sr)
    except Exception as e:
        raise type(e)("Error: Cannot segment the audio {} (onset: {}, duration: {})\n"
                      "Base Exception: {}".format(input_audio, onset, duration, e))
