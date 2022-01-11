import os
import sys
import argparse
import shutil
import pickle
from multiprocessing import Pool
from multiprocessing import current_process
from functools import partial
from pprint import pprint

import torch
import numpy as np
import tqdm

from model import load_model, CLASS_NAMES
from htk import HTKFile
from utils import seg_audio, extract_feature, read_text_file, dump_text_file, find_all_files, get_raw_filename

SEP = ' '
AUDIO_EXTENSION = ".wav"
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

def _run_vcm_rttm(vcm_model, smilextract_bin_path, input_audio_path, input_rttm_path, output_vcm_path,
                  all_children, keep_other, reuse_temp, keep_temp, skip_done):
    # Set up output filename
    if output_vcm_path is None:
        assert input_rttm_path.endswith('.rttm')
        output_vcm_path = input_rttm_path.replace('.rttm', '.vcm')
    elif os.path.isdir(output_vcm_path):
        output_vcm_path = os.path.join(output_vcm_path, os.path.basename(input_rttm_path).replace('.rttm', '.vcm'))

    # Should we skip the current file or not?
    if skip_done and os.path.exists(output_vcm_path):
        return

    # Used to store vcm predictions
    vcm_predictions = []
    input_rttm_data = read_text_file(input_rttm_path)

    # TQDM bar configuration
    bar_pos = current_process()._identity[0] + 1
    bar_des = '{}) {}'.format(bar_pos-1, get_raw_filename(input_rttm_path))

    # Process RTTM
    CHI_PATTERN = "CHI" if all_children else "KCHI"
    for line in tqdm.tqdm(input_rttm_data, position=bar_pos, desc=bar_des, leave=False):
        line = line.strip().split()

        file_name, onset, duration, speaker_type = line[1], line[3], line[4], line[7]

        # Do VCM prediction for children only
        if CHI_PATTERN not in speaker_type:
            if keep_other:
                vcm_predictions.append(SEP.join(line))
            continue

        temp_audio_filename = '{}_{}_{}{}'.format(file_name, onset, duration, AUDIO_EXTENSION)
        temp_audio_path = os.path.join(TMP_DIR, temp_audio_filename)
        temp_feature_path = temp_audio_path.replace(AUDIO_EXTENSION, '.htk') if AUDIO_EXTENSION != '' else \
                            temp_audio_path + '.htk'

        # If the number of keys is greater than one, then we are processing a directory for which we have listed
        # the audiofile. We only need find one that matches the filename specified in the RTTM file.
        if len(input_audio_path.keys()) > 1:
            assert file_name in input_audio_path.keys(), \
                'Error: audio file {} specified in RTTM {} not found!'.format(file_name, input_rttm_path)
            final_input_audio_path = input_audio_path[file_name]
        # Otherwise, the user wants to process a specific file with a specific RTTM, then the audio file is necessarily
        # the right one. Why do that? Because sometimes the filename in the RTTM file and the corresponding wavefile
        # is different
        else:
            final_input_audio_path = input_audio_path[list(input_audio_path.keys())[0]]

        # Segment audio file
        try:
            if not os.path.exists(temp_audio_path) or not reuse_temp:
                seg_audio(final_input_audio_path, temp_audio_path, onset, duration)
        except Exception as e:
            exit("Error: Cannot segment the audio {} for file {}"
                  "(onset: {}, duration: {})\n"
                 "Exception: {}".format(final_input_audio_path, file_name, onset, duration, e))

        # Extract features
        if not os.path.exists(temp_feature_path) or not reuse_temp:
            feature_rc, feature_stdout, feature_stderr = extract_feature(temp_audio_path,
                                                                         temp_feature_path,
                                                                         smilextract_bin_path)
            assert feature_rc == 0, 'OpenSMILE SMILExtract returned a non-zero ' \
                                    'exit code for file {}!\n{}'.format(file_name, feature_stderr)
            assert os.path.isfile(temp_feature_path), "Error: Feature file {} for file {} was " \
                                                      "not generated properly!".format(temp_feature_path, file_name)

        # Predict VCM
        try:
            vcm_prediction, vcm_confidence = predict_vcm(vcm_model, temp_feature_path, MEAN_VAR)
        except Exception as e:
            exit("Error: Cannot proceed with VCM prediction for file {} on: {}\n"
                 "Exception: {}".format(file_name, temp_audio_path, e))

        # Append VCM prediction
        line = LINE_PATTERN.format(file_name, onset, duration, vcm_prediction, float(vcm_confidence))
        vcm_predictions.append(line)

        # Remove temporary files
        if not keep_temp:
            os.remove(temp_audio_path)
            os.remove(temp_feature_path)

    assert not keep_other or len(input_rttm_data) == len(vcm_predictions), \
        "Error: Size mismatch for file {} (--keep-other={})" \
        "! Expected {}, got {}.".format(file_name, keep_other, len(input_rttm_data), len(vcm_predictions))

    # Dump predictions
    dump_text_file(output_vcm_path, vcm_predictions)

def _run_vcm_rttm_wrapper(input_rttm_path, **kwargs):
    try:
        _run_vcm_rttm(input_rttm_path=input_rttm_path, **kwargs)
    except Exception as e:
        log_fn = os.path.dirname(input_rttm_path).strip(os.sep).replace(os.sep, '-')
        with open('./log_{}.log'.format(log_fn), 'a+') as out_log_file:
            out_log_file.write('{}\n'.format(str(e)))
        return 1
    return 0

def run_vcm(smilextract_bin_path, input_audio_path, input_rttm_path, output_vcm_path, keep_temp, n_jobs, **kwargs):
    # Create temporary directory in VCM directory
    os.makedirs(TMP_DIR, exist_ok=True)

    # Check that the configuration files/directories we need exist
    assert os.path.exists(TMP_DIR), 'Temporary directory {} not found.'.format(TMP_DIR)
    assert os.path.isfile(MEAN_VAR), '{} not found (required by VCM model)'.format(MEAN_VAR)
    assert os.path.isfile(VCM_NET_MODEL_PATH), 'Pytorch model {} not found.'.format(VCM_NET_MODEL_PATH)
    assert os.access(smilextract_bin_path, os.X_OK), 'Path to OpenSMILE SMILExtract ({}) ' \
                                                     'is not executable!'.format(smilextract_bin_path)

    # The user can give either a path to a precise audio file or a path to a directory
    if os.path.isdir(input_audio_path):     # List recursively all the files in that directory
        audiofile_list = find_all_files(input_audio_path, AUDIO_EXTENSION)
    elif os.path.isfile(input_audio_path):  # Just one file
        audiofile_list = {get_raw_filename(input_audio_path): input_audio_path}
    else:  # We should not be getting here
        raise Exception("--input-audio-path is neither a file nor a directory.")

    # The user can give either a path to a precise RTTM file or a path to a directory
    if os.path.isdir(input_rttm_path):      # List recursively all the files in the directory
        rttmfile_list = [v for _, v in find_all_files(input_rttm_path, '.rttm').items()]
    elif os.path.isfile(input_rttm_path):   # Just one file
        rttmfile_list = [input_rttm_path]
    else: # We should not be getting here
        raise Exception("--input-rttm-path is neither a file nor a directory.")

    # Handle out directory/path: if the output directory does not exist: create it!
    if output_vcm_path is not None and not os.path.exists(output_vcm_path):
        os.makedirs(output_vcm_path, exist_ok=True)

    # Wrapped everything in a try/finally block to clear tmp_dir is something goes wrong
    try:
        # Load VCM model
        vcm_model = load_model(VCM_NET_MODEL_PATH)

        # Multiprocessing (does not work with multiple progress bars)
        with Pool(n_jobs) as p:
            args_dict = dict(vcm_model=vcm_model, smilextract_bin_path=smilextract_bin_path,
                        input_audio_path=audiofile_list, output_vcm_path=output_vcm_path, keep_temp=keep_temp)
            args_dict.update(**kwargs) # Add missing keys in kwargs
            f = partial(_run_vcm_rttm_wrapper, **args_dict)
            errors = list(tqdm.tqdm(p.imap_unordered(f, rttmfile_list), total=len(rttmfile_list), position=0))
            if errors: print('{} errors encountered. See log.'.format(sum(errors)))
    # Remove temporary directory
    finally:
        if not keep_temp: shutil.rmtree(TMP_DIR, ignore_errors=True)

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
    parser.add_argument("-x", "--audio-extension", default='.wav',
                        help="Audio files file extension (no extension '' also accepted). Default: '.wav'")
    parser.add_argument("-j", "-J", "--n-jobs", default=4, type=int,
                        help="Number of parallel jobs to run.")
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
    parser.add_argument("--skip-done", action='store_true',
                        help="Whether RTTM for which a VCM file already exists should be skipped. Default: False.")
    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    # Get arguments
    argv = sys.argv[1:]
    args = parse_arguments(argv)

    # Get arguments as dict
    args_dict = vars(args)

    # Add dot to the audio extension if forgotten by the user
    AUDIO_EXTENSION = args_dict.pop('audio_extension')
    if AUDIO_EXTENSION != '' and not AUDIO_EXTENSION.startswith('.'):
        AUDIO_EXTENSION = '.' + AUDIO_EXTENSION

    run_vcm(**args_dict)