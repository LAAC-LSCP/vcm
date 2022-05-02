import os
import shutil
from functools import partial

import torch
import tqdm
from torch.multiprocessing import Pool, current_process

from .model import load_model, predict_vcm
from .utils import RTTM_LINE_PATTERN, RTTM_SEP, _write_log, dump_text_file, extract_feature, find_all_files, \
    get_path_suffix, get_raw_filename, read_text_file, seg_audio

MEAN_VAR_PATH = os.path.join(os.path.dirname(__file__), '../config/vcm/vcm.eGeMAPS.func_utt.meanvar')
VCM_NET_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../config/model/vcm_model.pt')


def _run_vcm_rttm(vcm_model, smilextract_bin_path, input_audio_path, input_rttm_path_w_suffix,
                  output_vcm_path, tmp_dir, audio_extension, all_children=False, remove_others=False,
                  reuse_temp=False, keep_temp=False, skip_done=False, from_batched_vtc=False):
    input_rttm_path, input_rttm_path_suffix = input_rttm_path_w_suffix

    # Set up output filename
    if output_vcm_path is None:
        assert input_rttm_path.endswith('.rttm')
        output_vcm_path = input_rttm_path.replace('.rttm', '.vcm')
    elif os.path.isdir(output_vcm_path):
        os.makedirs(os.path.join(output_vcm_path, input_rttm_path_suffix), exist_ok=True)
        output_vcm_path = os.path.join(output_vcm_path, input_rttm_path_suffix,
                                       os.path.basename(input_rttm_path).replace('.rttm', '.vcm'))

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
            if not remove_others:
                vcm_predictions.append(RTTM_SEP.join(line))
            continue

        temp_audio_filename = '{}_{}_{}{}'.format(file_name, onset, duration, audio_extension)
        temp_audio_path = os.path.normpath(os.path.join(tmp_dir, temp_audio_filename))
        temp_feature_path = temp_audio_path.replace(audio_extension, '.htk') if audio_extension != '' else \
            temp_audio_path + '.htk'

        # If the number of keys is greater than one, then we are processing a directory for which we have listed
        # the audiofile. We only need find one that matches the filename specified in the RTTM file.
        if len(input_audio_path.keys()) > 1:
            file_name_key = '_'.join(file_name.split('_')[1:]) if from_batched_vtc else file_name
            assert file_name_key in input_audio_path.keys(), \
                'Error: audio file {} specified in RTTM {} not found!'.format(file_name_key, input_rttm_path)
            final_input_audio_path = input_audio_path[file_name_key]
        # Otherwise, the user wants to process a specific file with a specific RTTM, then the audio file is necessarily
        # the right one.
        else:
            final_input_audio_path = input_audio_path[list(input_audio_path.keys())[0]]

        # Segment audio file
        if not os.path.exists(temp_audio_path) or not reuse_temp:
            seg_audio(final_input_audio_path, temp_audio_path, onset, duration)

        # Extract features
        if not os.path.exists(temp_feature_path) or not reuse_temp:
            extract_feature(temp_audio_path, temp_feature_path, smilextract_bin_path)

        # Predict VCM
        vcm_prediction, vcm_confidence = predict_vcm(vcm_model, temp_feature_path, MEAN_VAR_PATH)

        # Append VCM prediction
        line = RTTM_LINE_PATTERN.format(file_name, onset, duration, vcm_prediction, float(vcm_confidence))
        vcm_predictions.append(line)

        # Remove temporary files
        if not keep_temp:
            os.remove(temp_audio_path)
            os.remove(temp_feature_path)

    assert remove_others or len(input_rttm_data) == len(vcm_predictions), \
        "Error: Size mismatch for file {} (--remove-others={})! " \
        "Expected {}, got {}.".format(file_name, remove_others, len(input_rttm_data), len(vcm_predictions))

    # Dump predictions
    dump_text_file(output_vcm_path, vcm_predictions)

def _run_vcm_rttm_wrapper(input_rttm_path, **kwargs):
    try:
        _run_vcm_rttm(input_rttm_path_w_suffix=input_rttm_path, **kwargs)
    except Exception as e:
        return str(e)
    return 0


def run_vcm(smilextract_bin_path, input_audio_path, input_rttm_path,
            output_vcm_path=None, audio_extension='.wav', keep_temp=False,
            n_jobs=4, temp_dir=None, **kwargs):
    # Add dot to the audio extension if forgotten by the user
    if audio_extension != '' and not audio_extension.startswith('.'):
        audio_extension = '.' + audio_extension

    # Normalise paths
    smilextract_bin_path = os.path.normpath(smilextract_bin_path)
    input_audio_path = os.path.normpath(input_audio_path)
    input_rttm_path = os.path.normpath(input_rttm_path)

    # Create temporary directory in VCM directory
    tmp_dir_suffix = input_rttm_path.strip(os.sep).replace(os.sep, '-').replace('.rttm', '')
    if temp_dir == None:
        temp_dir = os.path.join(os.path.dirname(__file__), '../tmp')
    tmp_dir = os.path.join(temp_dir, tmp_dir_suffix)
    try:
        os.makedirs(tmp_dir, exist_ok=True)
    except:
        raise IOError('Could not create temporary directory @ {}. '
                      'Try setting another path with --temp-dir.'.format(temp_dir))

    # Check that the configuration files/directories we need exist
    assert os.path.exists(tmp_dir), 'Temporary directory {} not found.'.format(tmp_dir)
    assert os.path.isfile(MEAN_VAR_PATH), '{} not found (required by VCM model)'.format(MEAN_VAR_PATH)
    assert os.path.isfile(VCM_NET_MODEL_PATH), 'Pytorch model {} not found.'.format(VCM_NET_MODEL_PATH)
    assert os.access(smilextract_bin_path, os.X_OK), 'Path to OpenSMILE SMILExtract ({}) ' \
                                                     'is not executable!'.format(smilextract_bin_path)

    # The user can give either a path to a precise audio file or a path to a directory
    if os.path.isdir(input_audio_path):  # List recursively all the files in that directory
        audiofile_list = find_all_files(input_audio_path, audio_extension)
    elif os.path.isfile(input_audio_path):  # Just one file
        audiofile_list = {get_raw_filename(input_audio_path): input_audio_path}
    else:  # We should not be getting here
        raise Exception("--input-audio-path is neither a file nor a directory.")

    # The user can give either a path to a precise RTTM file or a path to a directory
    # Structure [(/path/to/rttm/file.rttm, suffix/path w.r.t. input_rttm_path)]
    if os.path.isdir(input_rttm_path):  # List recursively all the files in the directory
        # Keep full path + path suffix to be able to reconstruct RTTM directory in output path
        rttmfile_list = [(v, get_path_suffix(v, input_rttm_path))  # full path, suffix
                         for _, v in find_all_files(input_rttm_path, '.rttm').items()]
    elif os.path.isfile(input_rttm_path):  # Just one file
        rttmfile_list = [(input_rttm_path, '')]
    else:  # We should not be getting here
        raise Exception("--input-rttm-path is neither a file nor a directory.")

    # Handle out directory/file path: if the output directory does not exist: create it!
    if output_vcm_path is not None:
        output_vcm_path = os.path.normpath(output_vcm_path)
        extension = os.path.splitext(output_vcm_path)[-1]
        # User specified a directory
        if extension == '' and not os.path.exists(output_vcm_path):
            os.makedirs(output_vcm_path, exist_ok=True)
        # User specified a path to a specific filename
        if extension != '' and not os.path.exists(os.path.dirname(output_vcm_path)):
            assert os.path.isfile(input_audio_path), \
                'Error: Can only use specific output file name if there is only one audio file to be processed!'
            os.makedirs(os.path.dirname(output_vcm_path), exist_ok=True)

    # Wrapped everything in a try/finally block to clear tmp_dir if something goes wrong
    try:
        # Load VCM model
        vcm_model = load_model(VCM_NET_MODEL_PATH)

        # Multiprocessing
        with Pool(n_jobs) as p:
            args_dict = dict(vcm_model=vcm_model, smilextract_bin_path=smilextract_bin_path,
                             input_audio_path=audiofile_list, output_vcm_path=output_vcm_path, keep_temp=keep_temp,
                             tmp_dir=tmp_dir, audio_extension=audio_extension)
            args_dict.update(**kwargs)  # Add missing keys in kwargs
            f = partial(_run_vcm_rttm_wrapper, **args_dict)
            errors = list(tqdm.tqdm(p.imap_unordered(f, rttmfile_list), total=len(rttmfile_list), position=0))
            errors_flt = list(filter(lambda e: type(e) == str, errors))
            if errors_flt:
                print('{} errors encountered. See log @ {}.'.format(len(errors_flt),
                                                                    _write_log(errors_flt, input_rttm_path)))
    # Remove temporary directory
    finally:
        if not keep_temp:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main(**kwargs):
    run_vcm(**kwargs)


def _parse_arguments(argv):
    import argparse

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("-a", "--input-audio-path", required=True,
                        help="Path to the audio file to be processed.")
    parser.add_argument("-r", "--input-rttm-path", required=True,
                        help="Path to the VTC output of the file to be processed.")
    parser.add_argument("-s", "--smilextract-bin-path", required=True,
                        help="Path to smilextract SMILExtract (v2.3) binary.")

    # Optional arguments
    parser.add_argument("-o", "--output-vcm-path", required=False,
                        help="Output path were the results of the VCM should be stored. Default: Same as RTTM file.")
    parser.add_argument("-x", "--audio-extension", required=False, default='.wav',
                        help="Audio files file extension (no extension '' also accepted). Default: '.wav'")

    # Conf. VCM output
    parser.add_argument("--all-children", action='store_true', required=False, default=False,
                        help="Should speech segment produced by other children than the key child (KCHI)"
                             "should be analysed. (Default: False.)")
    parser.add_argument("--remove-others", action='store_true', required=False, default=False,
                        help="Should the VTC annotations for the other speakers should be removed from the VCM"
                             "output file. If Segments from speaker-type SPEECH, MAL, FEM, etc. will be removed. "
                             "(Default: False.)")
    parser.add_argument("--from-batched-vtc", action='store_true', required=False, default=False,
                        help='Whether the VTC files were generated using LSCP/LAAC batch-voice-type-classifier or not.'
                             '/!\ LSCP/LAAC specific, you shouldn\'t be needing this option. (Default: False.)')

    # Temporary directory
    parser.add_argument("--keep-temp", action='store_true', required=False, default=False,
                        help="Whether temporary file should be kept or not. (Default: False.)")
    parser.add_argument("--reuse-temp", action='store_true', required=False, default=False,
                        help="Whether temporary file should be reused instead of being recomputed. (Default: False.)")
    parser.add_argument("--temp-dir", action='store', required=False, default=None,
                        help="Set path to temporary directory. (Default: `../tmp`).")
    parser.add_argument("--skip-done", action='store_true', required=False, default=False,
                        help="Whether RTTM for which a VCM file already exists should be skipped. (Default: False.)")

    # Multiprocessing configuration
    parser.add_argument("-j", "-J", "--n-jobs", required=False, default=4, type=int,
                        help="Number of parallel jobs to run.")

    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    import sys

    # Get argv arguments
    argv = sys.argv[1:]
    args = _parse_arguments(argv)

    args_dict = vars(args)
    main(**args_dict)
