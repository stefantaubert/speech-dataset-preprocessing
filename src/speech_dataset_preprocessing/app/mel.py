from functools import partial
from logging import getLogger
from multiprocessing import cpu_count
from pathlib import Path
from shutil import rmtree
from typing import Dict, Optional

import torch
from general_utils import get_chunk_name, load_obj, save_obj
from speech_dataset_preprocessing.app.ds import get_ds_dir
from speech_dataset_preprocessing.app.wav import get_wav_dir, load_wav_data
from speech_dataset_preprocessing.core.mel import MelDataList, process
from speech_dataset_preprocessing.core.wav import WavData
from speech_dataset_preprocessing.globals import DEFAULT_PRE_CHUNK_SIZE
from torch import Tensor

MEL_DATA_CSV = "data.pkl"


def __get_mel_root_dir(ds_dir: Path) -> Path:
  return ds_dir / "mel"


def get_mel_dir(ds_dir: Path, mel_name: str) -> Path:
  return __get_mel_root_dir(ds_dir) / mel_name


def load_mel_data(mel_dir: Path) -> MelDataList:
  path = mel_dir / MEL_DATA_CSV
  return load_obj(path)


def save_mel_data(mel_dir: Path, mel_data: MelDataList) -> None:
  path = mel_dir / MEL_DATA_CSV
  save_obj(mel_data, path)


def save_mel(dest_dir: Path, data_len: int, wav_entry: WavData, mel_tensor: Tensor) -> str:
  chunk_dir_name = get_chunk_name(
    i=wav_entry.entry_id,
    chunksize=DEFAULT_PRE_CHUNK_SIZE,
    maximum=data_len - 1
  )
  relative_dest_mel_path = Path(chunk_dir_name) / f"{wav_entry.entry_id}.pt"
  absolute_chunk_dir = dest_dir / chunk_dir_name
  absolute_dest_mel_path = dest_dir / relative_dest_mel_path

  absolute_chunk_dir.mkdir(parents=True, exist_ok=True)
  torch.save(mel_tensor, absolute_dest_mel_path)

  return relative_dest_mel_path


def preprocess_mels(base_dir: Path, ds_name: str, wav_name: str, custom_hparams: Optional[Dict[str, str]] = None, overwrite: bool = False):
  logger = getLogger(__name__)
  logger.info("Preprocessing mels...")
  ds_dir = get_ds_dir(base_dir, ds_name)
  mel_dir = get_mel_dir(ds_dir, wav_name)
  if mel_dir.is_dir() and not overwrite:
    logger.info("Already exists.")
    return

  wav_dir = get_wav_dir(ds_dir, wav_name)
  assert wav_dir.is_dir()
  data = load_wav_data(wav_dir)
  if len(data) == 0:
    return

  if mel_dir.is_dir():
    assert overwrite
    logger.info("Overwriting existing data.")
    rmtree(mel_dir)
  mel_dir.mkdir(exist_ok=False, parents=True)

  save_callback = partial(save_mel, dest_dir=mel_dir, data_len=len(data))
  mel_data = process(data, wav_dir, custom_hparams, save_callback, n_jobs=cpu_count() - 1)
  save_mel_data(mel_dir, mel_data)
  logger.info("Done.")
