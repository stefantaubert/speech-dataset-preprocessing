import os
from argparse import ArgumentParser
from typing import Dict, Optional

from text_utils import EngToIpaMode

from speech_dataset_preprocessing.app.ds import (preprocess_arctic,
                                                 preprocess_custom,
                                                 preprocess_libritts,
                                                 preprocess_ljs,
                                                 preprocess_mailabs,
                                                 preprocess_thchs,
                                                 preprocess_thchs_kaldi)
from speech_dataset_preprocessing.app.mel import preprocess_mels
from speech_dataset_preprocessing.app.plots import plot_mels
from speech_dataset_preprocessing.app.text import (preprocess_text,
                                                   text_convert_to_ipa,
                                                   text_normalize)
from speech_dataset_preprocessing.app.tools import remove_silence_plot
from speech_dataset_preprocessing.app.wav import (preprocess_wavs,
                                                  wavs_normalize,
                                                  wavs_remove_silence,
                                                  wavs_resample, wavs_stats,
                                                  wavs_stereo_to_mono)


def split_hparams_string(hparams: Optional[str]) -> Optional[Dict[str, str]]:
  if hparams is None:
    return None

  assignments = hparams.split(",")
  result = dict([x.split("=") for x in assignments])
  return result


def init_preprocess_thchs_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='THCHS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--ds_name', type=str, required=True, default='thchs')
  return preprocess_thchs


def init_preprocess_ljs_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='LJS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--ds_name', type=str, required=True, default='ljs')
  return preprocess_ljs


def init_preprocess_mailabs_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='M-AILABS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--ds_name', type=str, required=True, default='mailabs')
  return preprocess_mailabs


def init_preprocess_arctic_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='L2 Arctic dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--ds_name', type=str, required=True, default='arctic')
  return preprocess_arctic


def init_preprocess_libritts_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='LibriTTS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--ds_name', type=str, required=True, default='libritts')
  return preprocess_libritts


def init_preprocess_custom_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='LibriTTS dataset directory')
  parser.add_argument('--ds_name', type=str, required=True, default='custom')
  parser.set_defaults(auto_dl=False)
  return preprocess_custom


def init_preprocess_thchs_kaldi_parser(parser: ArgumentParser):
  parser.add_argument('--path', type=str, required=True, help='THCHS dataset directory')
  parser.add_argument('--auto_dl', action="store_true")
  parser.add_argument('--ds_name', type=str, required=True, default='thchs_kaldi')
  return preprocess_thchs_kaldi


def init_preprocess_mels_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--wav_name', type=str, required=True)
  parser.add_argument('--custom_hparams', type=str)
  return preprocess_mels_cli


def preprocess_mels_cli(**args):
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  preprocess_mels(**args)


def init_mels_plot_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--wav_name', type=str, required=True)
  parser.add_argument('--custom_hparams', type=str)
  return plot_mels_cli


def plot_mels_cli(**args):
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  plot_mels(**args)


def init_preprocess_text_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  return preprocess_text


def init_text_normalize_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--orig_text_name', type=str, required=True)
  parser.add_argument('--dest_text_name', type=str, required=True)
  return text_normalize


def init_text_convert_to_ipa_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--orig_text_name', type=str, required=True)
  parser.add_argument('--dest_text_name', type=str, required=True)
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--ignore_arcs', action='store_true')
  parser.add_argument('--consider_ipa_annotations', action='store_true')
  parser.add_argument('--mode', choices=EngToIpaMode,
                      type=EngToIpaMode.__getitem__)
  return text_convert_to_ipa


def init_preprocess_wavs_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--wav_name', type=str, required=True)
  return preprocess_wavs


def init_wavs_stats_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--wav_name', type=str, required=True)
  return wavs_stats


def init_wavs_normalize_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--orig_wav_name', type=str, required=True)
  parser.add_argument('--dest_wav_name', type=str, required=True)
  return wavs_normalize


def init_wavs_upsample_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--orig_wav_name', type=str, required=True)
  parser.add_argument('--dest_wav_name', type=str, required=True)
  parser.add_argument('--rate', type=int, required=True)
  return wavs_resample


def init_wavs_stereo_to_mono_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--orig_wav_name', type=str, required=True)
  parser.add_argument('--dest_wav_name', type=str, required=True)
  return wavs_stereo_to_mono


def init_wavs_remove_silence_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--orig_wav_name', type=str, required=True)
  parser.add_argument('--dest_wav_name', type=str, required=True)
  parser.add_argument('--chunk_size', type=int, required=True)
  parser.add_argument('--threshold_start', type=float, required=True)
  parser.add_argument('--threshold_end', type=float, required=True)
  parser.add_argument('--buffer_start_ms', type=float,
                      help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  parser.add_argument('--buffer_end_ms', type=float,
                      help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  return wavs_remove_silence


def init_wavs_remove_silence_plot_parser(parser: ArgumentParser):
  parser.add_argument('--ds_name', type=str, required=True)
  parser.add_argument('--wav_name', type=str, required=True)
  parser.add_argument('--chunk_size', type=int, required=True)
  parser.add_argument('--entry_id', type=int, help="Keep empty for random entry.")
  parser.add_argument('--threshold_start', type=float, required=True)
  parser.add_argument('--threshold_end', type=float, required=True)
  parser.add_argument('--buffer_start_ms', type=float,
                      help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  parser.add_argument('--buffer_end_ms', type=float,
                      help="amount of factors of chunk_size at the beginning and the end should be reserved", required=True)
  return remove_silence_plot


BASE_DIR_VAR = "base_dir"


def add_base_dir(parser: ArgumentParser):
  assert BASE_DIR_VAR in os.environ.keys()
  base_dir = os.environ[BASE_DIR_VAR]
  parser.set_defaults(base_dir=base_dir)


def _add_parser_to(subparsers, name: str, init_method):
  parser = subparsers.add_parser(name, help=f"{name} help")
  invoke_method = init_method(parser)
  parser.set_defaults(invoke_handler=invoke_method)
  add_base_dir(parser)
  return parser


def _init_parser():
  result = ArgumentParser()
  subparsers = result.add_subparsers(help='sub-command help')

  _add_parser_to(subparsers, "preprocess-custom", init_preprocess_custom_parser)
  _add_parser_to(subparsers, "preprocess-ljs", init_preprocess_ljs_parser)
  _add_parser_to(subparsers, "preprocess-mailabs", init_preprocess_mailabs_parser)
  _add_parser_to(subparsers, "preprocess-arctic", init_preprocess_arctic_parser)
  _add_parser_to(subparsers, "preprocess-libritts", init_preprocess_libritts_parser)
  _add_parser_to(subparsers, "preprocess-thchs", init_preprocess_thchs_parser)
  _add_parser_to(subparsers, "preprocess-thchs-kaldi", init_preprocess_thchs_kaldi_parser)

  _add_parser_to(subparsers, "preprocess-wavs", init_preprocess_wavs_parser)
  _add_parser_to(subparsers, "wavs-stats", init_wavs_stats_parser)
  _add_parser_to(subparsers, "wavs-normalize", init_wavs_normalize_parser)
  _add_parser_to(subparsers, "wavs-resample", init_wavs_upsample_parser)
  _add_parser_to(subparsers, "wavs-stereo-to-mono", init_wavs_stereo_to_mono_parser)
  _add_parser_to(subparsers, "wavs-remove-silence", init_wavs_remove_silence_parser)
  _add_parser_to(subparsers, "wavs-remove-silence-plot", init_wavs_remove_silence_plot_parser)

  _add_parser_to(subparsers, "preprocess-text", init_preprocess_text_parser)
  _add_parser_to(subparsers, "text-normalize", init_text_normalize_parser)
  _add_parser_to(subparsers, "text-ipa", init_text_convert_to_ipa_parser)

  _add_parser_to(subparsers, "preprocess-mels", init_preprocess_mels_parser)
  # is also possible without preprocess mels first
  _add_parser_to(subparsers, "mels-plot", init_mels_plot_parser)

  return result


def _process_args(args):
  params = vars(args)
  invoke_handler = params.pop("invoke_handler")
  invoke_handler(**params)


if __name__ == "__main__":
  main_parser = _init_parser()

  received_args = main_parser.parse_args()
  #args = main_parser.parse_args("ljs-text --base_dir=/datasets/models/taco2pt_v2 --mel_name=ljs --ds_name=test_ljs --convert_to_ipa".split())

  _process_args(received_args)
