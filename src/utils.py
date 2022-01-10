#!usr/bin/env python
# -*- coding: utf8 -*-

import os
from subprocess import PIPE, run
import librosa
import soundfile

def _clean(feature_output_path):
    if os.path.isfile(feature_output_path):
        os.remove(feature_output_path)


def read_text_file(input_path):
    with open(input_path) as file_in:
        return [line.strip() for line in file_in]


def dump_text_file(output_path, lines):
    with open(output_path, 'w') as file_out:
        file_out.write('\n'.join(lines))


def extract_feature(audio_input_path, feature_output_path, SMILEXTRACT_PATH):
    config = os.path.join(os.path.dirname(__file__), '../config/gemaps/eGeMAPSv01a.conf')

    cmd = f"{SMILEXTRACT_PATH} -C {config} -I {audio_input_path} -htkoutput {feature_output_path} -nologfile 1 >& /dev/null"
    command = cmd.split(' ')

    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)

    return result.returncode, result.stdout, result.stderr


def _seg_audio(input_audio, output_audio, onset, duration):
    # Original code
    assert os.path.isfile(input_audio)

    cmd_seg = "sox " + input_audio + " " + output_audio + " trim " + " " + onset + " " + duration
    subprocess.call(cmd_seg, shell=True)


def seg_audio(input_audio, output_audio, onset, duration):
    # onset and duration should be values provided in SECONDS
    wave_data, sr = librosa.load(input_audio, sr=None, offset=float(onset), duration=float(duration))
    soundfile.write(output_audio, wave_data, sr)