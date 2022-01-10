#!usr/bin/env python
# -*- coding: utf8 -*-

import os
import tqdm
from subprocess import PIPE, run
import librosa
import soundfile

#
# General utility functions
#

def _clean(feature_output_path):
    if os.path.isfile(feature_output_path):
        os.remove(feature_output_path)

def get_raw_filename(path):
    return os.path.splitext(os.path.basename(path))[0]

def read_text_file(input_path):
    with open(input_path) as file_in:
        return [line.strip() for line in file_in]


def dump_text_file(output_path, lines):
    with open(output_path, 'w') as file_out:
        file_out.write('\n'.join(lines))


def find_all_files(path, extension=''):
    audiofile_name2path = {}

    for p, d, f in os.walk(path):
        for file in f:
            if extension != '':
                if file.endswith(extension):
                    audiofile_name2path[get_raw_filename(file)] = os.path.join(p, file)
            else:
                audiofile_name2path[get_raw_filename(file)] = os.path.join(p, file)
    return audiofile_name2path

#
# VCM utility functions
#

def extract_feature(audio_input_path, feature_output_path, SMILEXTRACT_PATH):
    config = os.path.join(os.path.dirname(__file__), '../config/gemaps/eGeMAPSv01a.conf')

    cmd = f"{SMILEXTRACT_PATH} -C {config} -I {audio_input_path} -htkoutput {feature_output_path} -nologfile 1 >& /dev/null"
    command = cmd.split(' ')

    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)

    return result.returncode, result.stdout, result.stderr

def seg_audio(input_audio, output_audio, onset, duration):
    # onset and duration should be values provided in SECONDS
    wave_data, sr = librosa.load(input_audio, sr=None, offset=float(onset), duration=float(duration))
    soundfile.write(output_audio, wave_data, sr)