import os
from functools import partial
from pathlib import Path
from typing import Dict, Optional

import torch
from speech_dataset_preprocessing.app.ds import get_ds_dir
from speech_dataset_preprocessing.app.wav import get_wav_dir, load_wav_data
from speech_dataset_preprocessing.core.mel import MelData, MelDataList, process
from speech_dataset_preprocessing.core.wav import WavData
from speech_dataset_preprocessing.globals import DEFAULT_PRE_CHUNK_SIZE
from speech_dataset_preprocessing.utils import (get_chunk_name,
                                                get_pytorch_filename,
                                                get_subdir, load_obj, save_obj)
from torch import Tensor

MEL_DATA_CSV = "data.csv"


def _get_mel_root_dir(ds_dir: Path, create: bool = False) -> Path:
  return get_subdir(ds_dir, "mel", create)


def get_mel_dir(ds_dir: Path, mel_name: str, create: bool = False) -> Path:
  return get_subdir(_get_mel_root_dir(ds_dir, create), mel_name, create)


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
  relative_dest_wav_path = Path(chunk_dir_name) / get_pytorch_filename(repr(wav_entry))
  absolute_chunk_dir = dest_dir / chunk_dir_name
  absolute_dest_wav_path = dest_dir / relative_dest_wav_path

  os.makedirs(absolute_chunk_dir, exist_ok=True)
  torch.save(mel_tensor, absolute_dest_wav_path)

  return relative_dest_wav_path


def preprocess_mels(base_dir: Path, ds_name: str, wav_name: str, custom_hparams: Optional[Dict[str, str]] = None):
  print("Preprocessing mels...")
  ds_dir = get_ds_dir(base_dir, ds_name)
  mel_dir = get_mel_dir(ds_dir, wav_name)
  if mel_dir.is_dir():
    print("Already exists.")
  else:
    wav_dir = get_wav_dir(ds_dir, wav_name)
    assert wav_dir.is_dir()
    data = load_wav_data(wav_dir)
    assert len(data) > 0
    save_callback = partial(save_mel, dest_dir=mel_dir, data_len=len(data))
    mel_data = process(data, wav_dir, custom_hparams, save_callback)
    save_mel_data(mel_dir, mel_data)
