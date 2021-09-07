"""
input: wav data
output: mel data
"""
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Optional

from audio_utils.mel import TacotronSTFT, TSTFTHParams
from speech_dataset_preprocessing.core.wav import WavData, WavDataList
from speech_dataset_preprocessing.utils import (GenericList,
                                                overwrite_custom_hparams)
from torch import Tensor


@dataclass()
class MelData:
  entry_id: int
  relative_mel_path: Path
  n_mel_channels: int


class MelDataList(GenericList[MelData]):
  pass


def process(data: WavDataList, wav_dir: Path, custom_hparams: Optional[Dict[str, str]], save_callback: Callable[[WavData, Tensor], str]) -> MelDataList:
  result = MelDataList()
  hparams = TSTFTHParams()
  hparams = overwrite_custom_hparams(hparams, custom_hparams)
  mel_parser = TacotronSTFT(hparams, logger=getLogger())

  for wav_entry in data.items(True):
    absolute_wav_path = wav_dir / wav_entry.relative_wav_path
    mel_tensor = mel_parser.get_mel_tensor_from_file(absolute_wav_path)
    path = save_callback(wav_entry=wav_entry, mel_tensor=mel_tensor)
    mel_data = MelData(wav_entry.entry_id, path, hparams.n_mel_channels)
    result.append(mel_data)

  return result
