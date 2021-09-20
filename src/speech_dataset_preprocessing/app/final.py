from logging import getLogger
from pathlib import Path
from shutil import rmtree

from speech_dataset_preprocessing.app.ds import get_ds_dir, load_ds_data
from speech_dataset_preprocessing.app.mel import get_mel_dir, load_mel_data
from speech_dataset_preprocessing.app.text import get_text_dir, load_text_data
from speech_dataset_preprocessing.app.wav import get_wav_dir, load_wav_data
from speech_dataset_preprocessing.core.final import (FinalDsEntryList,
                                                     get_analysis_df,
                                                     get_final_ds_from_data)
from speech_dataset_preprocessing.globals import DEFAULT_CSV_SEPERATOR
from general_utils import load_obj, save_obj

FINAL_DATA_FILENAME = "data.pkl"
ANALYSIS_DF_FILENAME = "analysis.csv"


def save_final_ds(final_dir: Path, data: FinalDsEntryList) -> None:
  path = final_dir / FINAL_DATA_FILENAME
  save_obj(data, path)


def __load_final_ds(final_dir: Path) -> FinalDsEntryList:
  path = final_dir / FINAL_DATA_FILENAME
  return load_obj(path)


def load_final_ds(base_dir: Path, ds_name: str, final_name: Path) -> FinalDsEntryList:
  ds_dir = get_ds_dir(base_dir, ds_name)
  final_dir = get_final_dir(ds_dir, final_name)
  return __load_final_ds(final_dir)


def save_analysis_df(final_dir: Path, data: FinalDsEntryList) -> None:
  path = final_dir / ANALYSIS_DF_FILENAME
  df = get_analysis_df(data)
  df.to_csv(path, sep=DEFAULT_CSV_SEPERATOR, header=True, index=False)


def __get_final_root_dir(ds_dir: Path) -> Path:
  return ds_dir / "final"


def get_final_dir(ds_dir: Path, final_name: str) -> Path:
  return __get_final_root_dir(ds_dir) / final_name


def merge_to_final_ds(base_dir: Path, ds_name: str, text_name: str, audio_name: str, final_name: str, overwrite: bool) -> FinalDsEntryList:
  logger = getLogger(__name__)
  ds_dir = get_ds_dir(base_dir, ds_name)
  final_dir = get_final_dir(ds_dir, final_name)

  if final_dir.is_dir() and not overwrite:
    logger.info("Directory already exists!")
    return

  if not ds_dir.is_dir() or not ds_dir.exists():
    msg = "Dataset not found!"
    logger.exception(msg)
    raise Exception(msg)

  text_dir = get_text_dir(ds_dir, text_name)
  if not text_dir.is_dir() or not text_dir.exists():
    msg = "Text data not found!"
    logger.exception(msg)
    raise Exception(msg)

  wav_dir = get_wav_dir(ds_dir, audio_name)
  if not wav_dir.is_dir() or not wav_dir.exists():
    msg = "Wav data not found!"
    logger.exception(msg)
    raise Exception(msg)

  mel_dir = get_mel_dir(ds_dir, audio_name)
  if not mel_dir.is_dir() or not mel_dir.exists():
    msg = "Mel data not found!"
    logger.exception(msg)
    raise Exception(msg)

  ds_data = load_ds_data(ds_dir)
  text_data = load_text_data(text_dir)
  wav_data = load_wav_data(wav_dir)
  mel_data = load_mel_data(mel_dir)

  final_data = get_final_ds_from_data(
    ds_data=ds_data,
    text_data=text_data,
    wav_data=wav_data,
    mel_data=mel_data,
    wav_dir=wav_dir,
    mel_dir=mel_dir,
  )

  if final_dir.is_dir():
    assert overwrite
    logger.info("Overwriting existing data.")
    rmtree(final_dir)
  final_dir.mkdir(parents=True, exist_ok=False)

  save_final_ds(final_dir, final_data)
  save_analysis_df(final_dir, final_data)
  logger.info("Done.")
