import os
from logging import Logger, getLogger
from shutil import copyfile
from typing import Callable, Tuple

from speech_dataset_preprocessing.app.io import get_pre_dir
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
from text_utils import AccentsDict, SpeakersDict, SpeakersLogDict, SymbolIdDict
from unidecode import unidecode as convert_to_ascii

# don't do preprocessing here because inconsistent with mels because it is not always usefull to calc mels instand
# from speech_dataset_preprocessing.app.text import preprocess_text
# from speech_dataset_preprocessing.app.wav import preprocess_wavs
# from speech_dataset_preprocessing.app.mel import preprocess_mels

_ds_data_csv = "data.csv"
_ds_speakers_json = "speakers.json"
_ds_symbols_json = "symbols.json"
_ds_accents_json = "accents.json"


def _get_ds_root_dir(base_dir: str, create: bool = False):
  return get_subdir(get_pre_dir(base_dir, create), "ds", create)


def get_ds_dir(base_dir: str, ds_name: str, create: bool = False):
  return get_subdir(_get_ds_root_dir(base_dir, create), ds_name, create)


def get_ds_examples_dir(ds_dir: str, create: bool = False):
  return get_subdir(ds_dir, "examples", create)


def load_ds_csv(ds_dir: str) -> DsDataList:
  path = os.path.join(ds_dir, _ds_data_csv)
  res = DsDataList.load(DsData, path)
  return res


def _save_ds_csv(ds_dir: str, result: DsDataList):
  path = os.path.join(ds_dir, _ds_data_csv)
  result.save(path)


def load_symbols_json(ds_dir: str) -> SymbolIdDict:
  path = os.path.join(ds_dir, _ds_symbols_json)
  return SymbolIdDict.load_from_file(path)


def _save_symbols_json(ds_dir: str, data: SymbolIdDict):
  path = os.path.join(ds_dir, _ds_symbols_json)
  data.save(path)


def load_accents_json(ds_dir: str) -> AccentsDict:
  path = os.path.join(ds_dir, _ds_accents_json)
  return AccentsDict.load(path)


def _save_accents_json(ds_dir: str, data: AccentsDict):
  path = os.path.join(ds_dir, _ds_accents_json)
  data.save(path)


def load_speaker_json(ds_dir: str) -> SpeakersDict:
  path = os.path.join(ds_dir, _ds_speakers_json)
  return SpeakersDict.load(path)


def _save_speaker_json(ds_dir: str, speakers: SpeakersDict):
  path = os.path.join(ds_dir, _ds_speakers_json)
  speakers.save(path)


def _save_speaker_log_json(ds_dir: str, speakers_log: SpeakersLogDict):
  path = os.path.join(ds_dir, "speakers_log.json")
  speakers_log.save(path)


def _save_speaker_examples(ds_dir: str, examples: DsDataList, logger: Logger) -> None:
  logger.info("Saving examples for each speaker...")
  for example in examples.items(True):
    dest_file_name = f"{example.speaker_id}-{str(example.gender)}-{convert_to_ascii(example.speaker_name)}.wav"
    dest_path = os.path.join(get_ds_examples_dir(ds_dir, create=True), dest_file_name)
    copyfile(example.wav_path, dest_path)


def preprocess_thchs(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  logger = getLogger(__name__)
  logger.info("Preprocessing THCHS-30 dataset...")
  _preprocess_ds(base_dir, ds_name, path, auto_dl, thchs_preprocess, logger=logger)


def preprocess_thchs_kaldi(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  logger = getLogger(__name__)
  logger.info("Preprocessing THCHS-30 (Kaldi-Version) dataset...")
  _preprocess_ds(base_dir, ds_name, path, auto_dl, thchs_kaldi_preprocess, logger=logger)


def preprocess_ljs(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  logger = getLogger(__name__)
  logger.info("Preprocessing LJSpeech dataset...")
  _preprocess_ds(base_dir, ds_name, path, auto_dl, ljs_preprocess, logger=logger)


def preprocess_mailabs(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  logger = getLogger(__name__)
  logger.info("Preprocessing M-AILABS dataset...")
  _preprocess_ds(base_dir, ds_name, path, auto_dl, mailabs_preprocess, logger=logger)


def preprocess_libritts(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  logger = getLogger(__name__)
  logger.info("Preprocessing LibriTTS dataset...")
  _preprocess_ds(base_dir, ds_name, path, auto_dl, libritts_preprocess, logger=logger)


def preprocess_custom(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  logger = getLogger(__name__)
  logger.info("Preprocessing custom dataset...")
  _preprocess_ds(base_dir, ds_name, path, auto_dl, custom_preprocess, logger=logger)


def preprocess_arctic(base_dir: str, ds_name: str, path: str, auto_dl: bool):
  logger = getLogger(__name__)
  logger.info("Preprocessing L2 Arctic dataset...")
  _preprocess_ds(base_dir, ds_name, path, auto_dl, arctic_preprocess, logger=logger)


def _preprocess_ds(base_dir: str, ds_name: str, path: str, auto_dl: bool, preprocess_func: Callable[[str, bool, Logger], Tuple[
  SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict, AccentsDict]], logger: Logger):
  ds_dir = get_ds_dir(base_dir, ds_name, create=False)
  if os.path.isdir(ds_dir):
    logger.info("Dataset already processed.")
  else:
    logger.info("Reading data...")
    speakers, speakers_log, symbols, accents, ds_data = preprocess_func(path, auto_dl, logger)
    os.makedirs(ds_dir)
    _save_speaker_json(ds_dir, speakers)
    _save_speaker_log_json(ds_dir, speakers_log)
    _save_symbols_json(ds_dir, symbols)
    _save_accents_json(ds_dir, accents)
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
