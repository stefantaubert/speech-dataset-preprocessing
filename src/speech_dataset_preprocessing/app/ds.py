import os
from functools import partial
from logging import Logger, getLogger
from shutil import copyfile, rmtree
from typing import Callable, Tuple

from speech_dataset_preprocessing.core.ds import (DsData, DsDataList,
                                                  arctic_preprocess,
                                                  custom_preprocess,
                                                  get_speaker_examples,
                                                  libritts_preprocess,
                                                  ljs_preprocess,
                                                  mailabs_preprocess,
                                                  thchs_kaldi_preprocess,
                                                  thchs_preprocess)
from speech_dataset_preprocessing.utils import get_subdir
from text_utils import SpeakersDict, SpeakersLogDict, SymbolIdDict
from unidecode import unidecode as convert_to_ascii

# don't do preprocessing here because inconsistent with mels because it is not always usefull to calc mels instand
# from speech_dataset_preprocessing.app.text import preprocess_text
# from speech_dataset_preprocessing.app.wav import preprocess_wavs
# from speech_dataset_preprocessing.app.mel import preprocess_mels

_ds_data_csv = "data.csv"
_ds_speakers_json = "speakers.json"
_ds_symbols_json = "symbols.json"


def get_ds_dir(base_dir: str, ds_name: str, create: bool = False):
  return get_subdir(base_dir, ds_name, create)


def get_ds_examples_dir(ds_dir: str, create: bool = False):
  return get_subdir(ds_dir, "examples", create)


def load_ds_csv(ds_dir: str) -> DsDataList:
  path = os.path.join(ds_dir, _ds_data_csv)
  res = DsDataList.load(DsData, path)
  return res


def _save_ds_csv(ds_dir: str, result: DsDataList):
  path = os.path.join(ds_dir, _ds_data_csv)
  result.save(path)


def load_ds_symbols_json(ds_dir: str) -> SymbolIdDict:
  path = os.path.join(ds_dir, _ds_symbols_json)
  return SymbolIdDict.load_from_file(path)


def _save_ds_symbols_json(ds_dir: str, data: SymbolIdDict):
  path = os.path.join(ds_dir, _ds_symbols_json)
  data.save(path)


def load_ds_speaker_json(ds_dir: str) -> SpeakersDict:
  path = os.path.join(ds_dir, _ds_speakers_json)
  return SpeakersDict.load(path)


def _save_ds_speaker_json(ds_dir: str, speakers: SpeakersDict):
  path = os.path.join(ds_dir, _ds_speakers_json)
  speakers.save(path)


def _save_ds_speaker_log_json(ds_dir: str, speakers_log: SpeakersLogDict):
  path = os.path.join(ds_dir, "speakers_log.json")
  speakers_log.save(path)


def _save_speaker_examples(ds_dir: str, examples: DsDataList, logger: Logger) -> None:
  logger.info("Saving examples for each speaker...")
  for example in examples.items(True):
    dest_file_name = f"{example.speaker_id}-{str(example.gender)}-{convert_to_ascii(example.speaker_name)}.wav"
    dest_path = os.path.join(get_ds_examples_dir(ds_dir, create=True), dest_file_name)
    copyfile(example.wav_path, dest_path)


def preprocess_thchs(base_dir: str, ds_name: str, path: str, auto_dl: bool, overwrite: bool):
  logger = getLogger(__name__)
  logger.info("Preprocessing THCHS-30 dataset...")
  preprocess_func = partial(thchs_preprocess, dir_path=path, auto_dl=auto_dl)
  _preprocess_ds(base_dir, ds_name, preprocess_func, overwrite=overwrite)


def preprocess_thchs_kaldi(base_dir: str, ds_name: str, path: str, auto_dl: bool, overwrite: bool):
  logger = getLogger(__name__)
  logger.info("Preprocessing THCHS-30 (Kaldi-Version) dataset...")
  preprocess_func = partial(thchs_kaldi_preprocess, dir_path=path, auto_dl=auto_dl)
  _preprocess_ds(base_dir, ds_name, preprocess_func, overwrite=overwrite)


def preprocess_ljs(base_dir: str, ds_name: str, path: str, auto_dl: bool, overwrite: bool):
  logger = getLogger(__name__)
  logger.info("Preprocessing LJSpeech dataset...")
  preprocess_func = partial(ljs_preprocess, dir_path=path, auto_dl=auto_dl)
  _preprocess_ds(base_dir, ds_name, preprocess_func, overwrite=overwrite)


def preprocess_mailabs(base_dir: str, ds_name: str, path: str, auto_dl: bool, overwrite: bool):
  logger = getLogger(__name__)
  logger.info("Preprocessing M-AILABS dataset...")
  preprocess_func = partial(mailabs_preprocess, dir_path=path, auto_dl=auto_dl)
  _preprocess_ds(base_dir, ds_name, preprocess_func, overwrite=overwrite)


def preprocess_libritts(base_dir: str, ds_name: str, path: str, auto_dl: bool, overwrite: bool):
  logger = getLogger(__name__)
  logger.info("Preprocessing LibriTTS dataset...")
  preprocess_func = partial(libritts_preprocess, dir_path=path, auto_dl=auto_dl)
  _preprocess_ds(base_dir, ds_name, preprocess_func, overwrite=overwrite)


def preprocess_arctic(base_dir: str, ds_name: str, path: str, auto_dl: bool, overwrite: bool):
  logger = getLogger(__name__)
  logger.info("Preprocessing L2 Arctic dataset...")
  preprocess_func = partial(arctic_preprocess, dir_path=path, auto_dl=auto_dl)
  _preprocess_ds(base_dir, ds_name, preprocess_func, overwrite=overwrite)


def preprocess_custom(base_dir: str, ds_name: str, path: str, overwrite: bool):
  logger = getLogger(__name__)
  logger.info("Preprocessing custom dataset...")
  preprocess_func = partial(custom_preprocess, dir_path=path)
  _preprocess_ds(base_dir, ds_name, preprocess_func, overwrite=overwrite)


def _preprocess_ds(base_dir: str, ds_name: str, preprocess_func: Callable[[], Tuple[
  SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict]], overwrite: bool):
  ds_dir = get_ds_dir(base_dir, ds_name, create=False)
  logger = getLogger(__name__)
  if os.path.isdir(ds_dir) and not overwrite:
    logger.info("Dataset already processed.")
    return

  logger.info("Reading data...")
  speakers, speakers_log, symbols, ds_data = preprocess_func()
  if os.path.isdir(ds_dir):
    assert overwrite
    logger.info("Overwriting existing data.")
    rmtree(ds_dir)
  os.makedirs(ds_dir)
  _save_ds_speaker_json(ds_dir, speakers)
  _save_ds_speaker_log_json(ds_dir, speakers_log)
  _save_ds_symbols_json(ds_dir, symbols)
  _save_ds_csv(ds_dir, ds_data)
  examples = get_speaker_examples(ds_data)
  _save_speaker_examples(ds_dir, examples, logger)
  logger.info("Dataset processed.")


def add_speaker_examples(base_dir: str, ds_name: str):
  logger = getLogger(__name__)
  ds_dir = get_ds_dir(base_dir, ds_name, create=False)
  ds_data = load_ds_csv(ds_dir)
  examples = get_speaker_examples(ds_data)
  _save_speaker_examples(ds_dir, examples, logger)
