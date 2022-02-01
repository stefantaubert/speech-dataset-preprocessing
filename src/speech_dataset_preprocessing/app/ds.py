from functools import partial
from logging import Logger, getLogger
from pathlib import Path
from shutil import copyfile, rmtree
from typing import Callable

from general_utils import load_obj, save_obj
from speech_dataset_preprocessing.core.ds import (DsDataList,
                                                  PreprocessingResult,
                                                  arctic_preprocess,
                                                  generic_preprocess,
                                                  get_speaker_examples,
                                                  libritts_preprocess,
                                                  ljs_preprocess,
                                                  mailabs_preprocess,
                                                  thchs_kaldi_preprocess,
                                                  thchs_preprocess)
from text_utils import SpeakersLogDict, SymbolFormat
from unidecode import unidecode as convert_to_ascii

# don't do preprocessing here because inconsistent with mels because it is not always usefull to calc mels instand
# from speech_dataset_preprocessing.app.text import preprocess_text
# from speech_dataset_preprocessing.app.wav import preprocess_wavs
# from speech_dataset_preprocessing.app.mel import preprocess_mels

_ds_data_csv = "data.pkl"


def get_ds_dir(base_dir: Path, ds_name: str) -> Path:
  return base_dir / ds_name


def get_ds_examples_dir(ds_dir: Path) -> Path:
  return ds_dir / "examples"


def __save_ds_data(ds_dir: Path, result: DsDataList) -> None:
  path = ds_dir / _ds_data_csv
  save_obj(result, path)


def load_ds_data(ds_dir: Path) -> DsDataList:
  path = ds_dir / _ds_data_csv
  return load_obj(path)


def _save_ds_speaker_log_json(ds_dir: Path, speakers_log: SpeakersLogDict) -> None:
  path = ds_dir / "speakers_log.json"
  speakers_log.save(path)


def _save_speaker_examples(ds_dir: Path, examples: DsDataList, logger: Logger) -> None:
  logger.info("Saving examples for each speaker...")
  example_dir = get_ds_examples_dir(ds_dir)
  example_dir.mkdir(exist_ok=True, parents=True)
  for i, example in enumerate(examples.items(True), start=1):
    dest_file_name = f"{i}-{str(example.speaker_gender)}-{convert_to_ascii(example.speaker_name)}.wav"
    dest_path = example_dir / dest_file_name
    copyfile(example.wav_absolute_path, dest_path)


def preprocess_thchs(base_dir: Path, ds_name: str, path: Path, auto_dl: bool, overwrite: bool):
  logger = getLogger(__name__)
  logger.info("Preprocessing THCHS-30 dataset...")
  preprocess_func = partial(thchs_preprocess, dir_path=path, auto_dl=auto_dl)
  __preprocess_ds(base_dir, ds_name, preprocess_func, overwrite=overwrite)


def preprocess_thchs_kaldi(base_dir: Path, ds_name: str, path: Path, auto_dl: bool, overwrite: bool):
  logger = getLogger(__name__)
  logger.info("Preprocessing THCHS-30 (Kaldi-Version) dataset...")
  preprocess_func = partial(thchs_kaldi_preprocess, dir_path=path, auto_dl=auto_dl)
  __preprocess_ds(base_dir, ds_name, preprocess_func, overwrite=overwrite)


def preprocess_ljs(base_dir: Path, ds_name: str, path: Path, auto_dl: bool, overwrite: bool):
  logger = getLogger(__name__)
  logger.info("Preprocessing LJSpeech dataset...")
  preprocess_func = partial(ljs_preprocess, dir_path=path, auto_dl=auto_dl)
  __preprocess_ds(base_dir, ds_name, preprocess_func, overwrite=overwrite)


def preprocess_mailabs(base_dir: Path, ds_name: str, path: Path, overwrite: bool):
  logger = getLogger(__name__)
  logger.info("Preprocessing M-AILABS dataset...")
  preprocess_func = partial(mailabs_preprocess, dir_path=path)
  __preprocess_ds(base_dir, ds_name, preprocess_func, overwrite=overwrite)


def preprocess_libritts(base_dir: Path, ds_name: str, path: Path, overwrite: bool):
  logger = getLogger(__name__)
  logger.info("Preprocessing LibriTTS dataset...")
  preprocess_func = partial(libritts_preprocess, dir_path=path)
  __preprocess_ds(base_dir, ds_name, preprocess_func, overwrite=overwrite)


def preprocess_arctic(base_dir: Path, ds_name: str, path: Path, overwrite: bool):
  logger = getLogger(__name__)
  logger.info("Preprocessing L2 Arctic dataset...")
  preprocess_func = partial(arctic_preprocess, dir_path=path)
  __preprocess_ds(base_dir, ds_name, preprocess_func, overwrite=overwrite)


def preprocess_generic(base_dir: Path, ds_name: str, path: Path, tier_name: str, n_digits: int, symbols_format: SymbolFormat, overwrite: bool):
  logger = getLogger(__name__)
  logger.info("Preprocessing generic dataset...")
  preprocess_func = partial(
    generic_preprocess,
    directory=path,
    tier_name=tier_name,
    n_digits=n_digits,
    symbols_format=symbols_format
  )
  __preprocess_ds(base_dir, ds_name, preprocess_func, overwrite=overwrite)


def __preprocess_ds(base_dir: Path, ds_name: str, preprocess_func: Callable[[], PreprocessingResult], overwrite: bool):
  ds_dir = get_ds_dir(base_dir, ds_name)
  logger = getLogger(__name__)
  if ds_dir.is_dir() and not overwrite:
    logger.info("Dataset already processed.")
    return

  logger.info("Reading data...")
  speakers_log, ds_data = preprocess_func()
  if ds_dir.exists():
    assert overwrite
    logger.info("Overwriting existing data.")
    rmtree(ds_dir)
  ds_dir.mkdir(exist_ok=False, parents=True)

  _save_ds_speaker_log_json(ds_dir, speakers_log)
  __save_ds_data(ds_dir, ds_data)
  examples = get_speaker_examples(ds_data)
  _save_speaker_examples(ds_dir, examples, logger)
  logger.info("Dataset processed.")


def add_speaker_examples(base_dir: str, ds_name: str):
  logger = getLogger(__name__)
  ds_dir = get_ds_dir(base_dir, ds_name)
  ds_data = load_ds_data(ds_dir)
  examples = get_speaker_examples(ds_data)
  _save_speaker_examples(ds_dir, examples, logger)
