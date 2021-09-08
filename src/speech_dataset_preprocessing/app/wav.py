from functools import partial
from logging import getLogger
from pathlib import Path
from shutil import rmtree
from typing import Callable

from speech_dataset_preprocessing.app.ds import get_ds_dir, load_ds_data
from speech_dataset_preprocessing.core.wav import (WavDataList, log_stats,
                                                   normalize, preprocess,
                                                   remove_silence, resample,
                                                   stereo_to_mono)
from speech_dataset_preprocessing.utils import get_subdir, load_obj, save_obj

_wav_data_csv = "data.pkl"


def _get_wav_root_dir(ds_dir: Path, create: bool = False) -> Path:
  return get_subdir(ds_dir, "wav", create)


def get_wav_dir(ds_dir: Path, wav_name: str, create: bool = False) -> Path:
  return get_subdir(_get_wav_root_dir(ds_dir, create), wav_name, create)


def load_wav_data(wav_dir: Path) -> WavDataList:
  path = wav_dir / _wav_data_csv
  return load_obj(path)


def save_wav_data(wav_dir: Path, wav_data: WavDataList) -> None:
  wav_dir.mkdir(parents=True, exist_ok=True)
  path = wav_dir / _wav_data_csv
  save_obj(wav_data, path)


def preprocess_wavs(base_dir: Path, ds_name: str, wav_name: str, overwrite: bool = False) -> None:
  logger = getLogger(__name__)
  logger.info("Preprocessing wavs...")
  ds_dir = get_ds_dir(base_dir, ds_name)
  dest_wav_dir = get_wav_dir(ds_dir, wav_name)
  if dest_wav_dir.is_dir() and not overwrite:
    logger.error("Already exists.")
    return

  data = load_ds_data(ds_dir)

  if dest_wav_dir.is_dir():
    assert overwrite
    logger.info("Overwriting existing data.")
    rmtree(dest_wav_dir)
  dest_wav_dir.mkdir(exist_ok=False, parents=True)

  wav_data = preprocess(data, dest_wav_dir)
  save_wav_data(dest_wav_dir, wav_data)
  ds_data = load_ds_data(ds_dir)
  log_stats(ds_data, wav_data)


def wavs_stats(base_dir: Path, ds_name: str, wav_name: str) -> None:
  logger = getLogger(__name__)
  logger.info(f"Stats of {wav_name}")
  ds_dir = get_ds_dir(base_dir, ds_name)
  wav_dir = get_wav_dir(ds_dir, wav_name)
  if wav_dir.is_dir():
    ds_data = load_ds_data(ds_dir)
    wav_data = load_wav_data(wav_dir)
    log_stats(ds_data, wav_data)


def wavs_normalize(base_dir: Path, ds_name: str, orig_wav_name: str, dest_wav_name: str, overwrite: bool = False) -> None:
  logger = getLogger(__name__)
  logger.info("Normalizing wavs...")
  op = partial(normalize)
  __wav_op(base_dir, ds_name, orig_wav_name, dest_wav_name, op, overwrite)


def wavs_resample(base_dir: Path, ds_name: str, orig_wav_name: str, dest_wav_name: str, rate: int, overwrite: bool = False) -> None:
  logger = getLogger(__name__)
  logger.info("Resampling wavs...")
  op = partial(resample, new_rate=rate)
  __wav_op(base_dir, ds_name, orig_wav_name, dest_wav_name, op, overwrite)


def wavs_stereo_to_mono(base_dir: Path, ds_name: str, orig_wav_name: str, dest_wav_name: str, overwrite: bool = False) -> None:
  logger = getLogger(__name__)
  logger.info("Converting wavs from stereo to mono...")
  op = partial(stereo_to_mono)
  __wav_op(base_dir, ds_name, orig_wav_name, dest_wav_name, op, overwrite)


def wavs_remove_silence(base_dir: Path, ds_name: str, orig_wav_name: str, dest_wav_name: str, chunk_size: int, threshold_start: float, threshold_end: float, buffer_start_ms: float, buffer_end_ms: float, overwrite: bool = False) -> None:
  logger = getLogger(__name__)
  logger.info("Removing silence in wavs...")
  op = partial(remove_silence, chunk_size=chunk_size, threshold_start=threshold_start,
               threshold_end=threshold_end, buffer_start_ms=buffer_start_ms, buffer_end_ms=buffer_end_ms)
  __wav_op(base_dir, ds_name, orig_wav_name, dest_wav_name, op, overwrite)


def __wav_op(base_dir: Path, ds_name: str, origin_wav_name: str, destination_wav_name: str, op: Callable[[WavDataList, Path, Path], WavDataList], overwrite: bool) -> None:
  logger = getLogger(__name__)
  ds_dir = get_ds_dir(base_dir, ds_name)
  dest_wav_dir = get_wav_dir(ds_dir, destination_wav_name)
  if dest_wav_dir.is_dir() and not overwrite:
    logger.error("Already exists.")
    return

  orig_wav_dir = get_wav_dir(ds_dir, origin_wav_name)
  assert orig_wav_dir.is_dir()
  data = load_wav_data(orig_wav_dir)

  if dest_wav_dir.is_dir():
    assert overwrite
    logger.info("Overwriting existing data.")
    rmtree(dest_wav_dir)

  dest_wav_dir.mkdir(exist_ok=False, parents=True)
  wav_data = op(data, orig_wav_dir, dest_wav_dir)
  save_wav_data(dest_wav_dir, wav_data)
  ds_data = load_ds_data(ds_dir)
  log_stats(ds_data, wav_data)
