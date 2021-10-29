"""
input: wav data
output: mel data
"""
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Optional

from audio_utils.mel import TacotronSTFT, TSTFTHParams
from general_utils import GenericList, overwrite_custom_hparams
from speech_dataset_preprocessing.core.wav import WavData, WavDataList
from torch import Tensor
from tqdm import tqdm


@dataclass()
class MelData:
  entry_id: int
  mel_relative_path: Path
  mel_n_channels: int


class MelDataList(GenericList[MelData]):
  pass


def process_entry(entry: WavData, wav_dir: Path, mel_parser: TacotronSTFT, save_callback: Callable[[WavData, Tensor], str]) -> MelData:
  absolute_wav_path = wav_dir / entry.wav_relative_path
  mel_tensor = mel_parser.get_mel_tensor_from_file(absolute_wav_path)
  path = save_callback(wav_entry=entry, mel_tensor=mel_tensor)
  mel_data = MelData(entry.entry_id, path, mel_parser.n_mel_channels)
  return mel_data


def process(data: WavDataList, wav_dir: Path, custom_hparams: Optional[Dict[str, str]], save_callback: Callable[[WavData, Tensor], str], n_jobs: int) -> MelDataList:
  hparams = TSTFTHParams()
  hparams = overwrite_custom_hparams(hparams, custom_hparams)
  mel_parser = TacotronSTFT(hparams, logger=getLogger())
  mt_method = partial(
    process_entry,
    wav_dir=wav_dir,
    mel_parser=mel_parser,
    save_callback=save_callback,
  )

  with ThreadPoolExecutor(max_workers=n_jobs) as ex:
    result = MelDataList(tqdm(ex.map(mt_method, data.items()), total=len(data)))

  return result
