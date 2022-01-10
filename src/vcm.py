import os
import sys
import argparse
import shutil
import pickle
import numpy as np
import tqdm

import torch

from model import load_model, CLASS_NAMES
from htk import HTKFile
from utils import seg_audio, extract_feature, read_text_file, dump_text_file

SEP = ' '
AUDIO_EXTENSION = "wav" # maybe use an argument to specify extension if there are some file that are not wave files
LINE_PATTERN =  "SPEAKER {} 1 {} {} <NA> <NA> {} {:.2f} <NA>".replace(' ', SEP) # fn, onset, duration, vcm-class, conf

TMP_DIR = os.path.join(os.path.dirname(__file__), '../tmp')
MEAN_VAR = os.path.join(os.path.dirname(__file__), '../config/vcm/vcm.eGeMAPS.func_utt.meanvar')
VCM_NET_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../config/model/vcm_model.pt')

def predict_vcm(model, input, mean_var):
    # Read normalisation parameters
    assert os.path.exists(mean_var)

    with open(mean_var, 'rb') as f:
        mv = pickle.load(f)

    m, v = mv['mean'], mv['var']
    std = lambda feat: (feat - m) / v

    # Load input feature and predict
    htk_reader = HTKFile()
    htk_reader.load(input)

    feat = std(np.array(htk_reader.data))
    input = torch.from_numpy(feat.astype('float32'))

    with torch.no_grad():
        output_ling = model(input).data.data.cpu().numpy()
    prediction_confidence = output_ling.max()  # post propability

    cls_ling = np.argmax(output_ling)
    predition_vcm = CLASS_NAMES[cls_ling]  # prediction

    return predition_vcm, prediction_confidence

def _run_vcm(smilextract_bin_path, input_audio_path, input_rttm_path, output_vcm_path,
            all_children=False, keep_other=False, reuse_temp=False, keep_temp=False):

    # Load VCM model
    vcm_model = load_model(VCM_NET_MODEL_PATH)

    vcm_predictions = []
    input_rttm_data = read_text_file(input_rttm_path)

    CHI_PATTERN = "CHI" if all_children else "KCHI"

    for line in tqdm.tqdm(input_rttm_data):
        line = line.strip().split()

        file_name, onset, duration, speaker_type = line[1], line[3], line[4], line[7]

        # Do VCM prediction for children only
        if CHI_PATTERN in speaker_type:
            temp_audio_filename = '{}_{}_{}.{}'.format(file_name, onset, duration, AUDIO_EXTENSION)
            temp_audio_path = os.path.join(TMP_DIR, temp_audio_filename)
            temp_feature_path = temp_audio_path.replace(AUDIO_EXTENSION, 'htk')

            # If input_rttm_path is all.rttm then input_audio_path is a directory
            # Hence, we should append the filename we expect to find in that directory
            if os.path.isdir(input_audio_path):
                final_input_audio_path = os.path.join(input_audio_path, temp_audio_filename)
            else:
                final_input_audio_path = input_audio_path
            assert os.path.isfile(final_input_audio_path), 'Error: audio file {} not found'.format(final_input_audio_path)

            # Segment audio file
            try:
                if not os.path.exists(temp_audio_path) or not reuse_temp:
                    seg_audio(final_input_audio_path, temp_audio_path, onset, duration)
            except Exception as e:
                exit("Error: Cannot segment the audio: {} "
                      "(onset: {}, duration: {})\n"
                     "Exception: {}".format(final_input_audio_path, onset, duration, e))

            # Extract features
            if not os.path.exists(temp_feature_path) or not reuse_temp:
                feature_rc, feature_stdout, feature_stderr = extract_feature(temp_audio_path,
                                                                             temp_feature_path,
                                                                             smilextract_bin_path)
                assert feature_rc == 0, 'OpenSMILE SMILExtract returned a non-zero exit code!\n{}'.format(feature_stderr)
                assert os.path.isfile(temp_feature_path), "Error: Feature file {} was " \
                                                          "not generated properly!".format(temp_feature_path)

            # Predict VCM
            try:
                vcm_prediction, vcm_confidence = predict_vcm(vcm_model, temp_feature_path, MEAN_VAR)
            except Exception as e:
                exit("Error: Cannot proceed vcm prediction on: {}\n"
                     "Exception: {}".format(temp_audio_path, e))

            # Append VCM prediction
            line = LINE_PATTERN.format(file_name, onset, duration, vcm_prediction, float(vcm_confidence))
            vcm_predictions.append(line)

            # Remove temporary files
            if not keep_temp:
                os.remove(temp_audio_path)
                os.remove(temp_feature_path)

        if keep_other and CHI_PATTERN not in speaker_type:
            vcm_predictions.append(SEP.join(line))

    assert not keep_other or len(input_rttm_data) == len(vcm_predictions), "Error: Size mismatch!"

    # Dump predictions
    dump_text_file(output_vcm_path, vcm_predictions)

def run_vcm(*args, keep_temp=False, **kwargs):
    # Create temporary directory in VCM directory
    os.makedirs(TMP_DIR, exist_ok=True)
    assert os.path.exists(TMP_DIR), 'Temporary directory {} not found.'.format(TMP_DIR)

    try:
        _run_vcm(*args, keep_temp=keep_temp, **kwargs)
    except Exception as e:
        exit(e)
    finally:
        # Remove temporary directory
        if not keep_temp:
            shutil.rmtree(TMP_DIR, ignore_errors=True)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--input-audio-path", required=True,
                        help="Path to the audio file to be processed.")
    parser.add_argument("-r", "--input-rttm-path", required=True,
                        help="Path to the VTC output of the file to be processed.")
    parser.add_argument("-s", "--smilextract-bin-path", required=True,
                        help="Path to smilextract SMILExtract (v2.3) binary.")
    parser.add_argument("-o", "--output-vcm-path",
                        help="Output path were the results of the VCM should be stored. Default: Same as RTTM file.")
    parser.add_argument("--all-children", action='store_true',
                        help="Should speech segment produced by other children than the key child (KCHI)"
                             "should be analysed. Default: False")
    parser.add_argument("--keep-other", action='store_true',
                        help="Should the VTC annotations for the other speakers should be transfered into the VCM"
                             "output file. Segments from speaker-type SPEECH, MAL, FEM, etc.) will be kept."
                             "Default: False.")
    parser.add_argument("--keep-temp", action='store_true',
                        help="Whether temporary file should be kept or not. Default: False.")
    parser.add_argument("--reuse-temp", action='store_true',
                        help="Whether temporary file should be reused instead of being recomputed. Default: False.")
    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    # Get arguments
    argv = sys.argv[1:]
    args = parse_arguments(argv)

    input_audio_path = args.input_audio_path
    input_rttm_path = args.input_rttm_path
    smilextract_bin_path = args.smilextract_bin_path
    all_children = args.all_children
    keep_other = args.keep_other
    keep_temp = args.keep_temp
    reuse_temp = args.reuse_temp

    # Create output directory if necessary
    if args.output_vcm_path is None:
        assert input_rttm_path.endswith('.rttm')
        output_vcm_path = input_rttm_path.replace('.rttm', '.vcm')
    else:
        if not os.path.exists(args.output_vcm_path): os.makedirs(args.output_vcm_path, exist_ok=True)
        output_vcm_path = os.path.join(args.output_vcm_path, os.path.basename(input_rttm_path))


    # Check that the provided path/files are okay
    assert os.path.isfile(input_rttm_path), 'Input RTTM file not found.'
    # If the RTTM file is a 'all.rttm' file we consider input_audio_path as a directory and not a file path
    if os.path.basename(input_rttm_path) != 'all.rttm':
        assert os.path.isfile(input_audio_path), 'Input audio file not found.'
    else:
        assert os.path.isdir(input_audio_path), 'Input audio path directory not found.'

    assert os.path.isfile(MEAN_VAR), '{} not found (required by VCM model)'.format(MEAN_VAR)
    assert os.path.isfile(VCM_NET_MODEL_PATH), 'Pytorch model {} not found.'.format(VCM_NET_MODEL_PATH)

    run_vcm(smilextract_bin_path, input_audio_path, input_rttm_path, output_vcm_path, all_children, keep_other,
            reuse_temp, keep_temp=keep_temp)