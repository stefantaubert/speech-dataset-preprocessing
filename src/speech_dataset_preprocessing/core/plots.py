"""
input: wav data
output: mel data
"""
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, List, Optional

from audio_utils.mel import TacotronSTFT, TSTFTHParams
from speech_dataset_preprocessing.core.ds import DsData, DsDataList
from speech_dataset_preprocessing.core.wav import WavData, WavDataList
from speech_dataset_preprocessing.utils import overwrite_custom_hparams


def process(data: WavDataList, ds: DsDataList, wav_dir: Path, custom_hparams: Optional[Dict[str, str]], save_callback: Callable[[WavData, DsData], Path]) -> List[Path]:
  hparams = TSTFTHParams()
  hparams = overwrite_custom_hparams(hparams, custom_hparams)
  mel_parser = TacotronSTFT(hparams, logger=getLogger())

  all_paths: List[Path] = []
  for wav_entry, ds_entry in zip(data.items(True), ds.items(True)):
    absolute_wav_path = wav_dir / wav_entry.relative_wav_path
    mel_tensor = mel_parser.get_mel_tensor_from_file(absolute_wav_path)
    absolute_path = save_callback(wav_entry=wav_entry, ds_entry=ds_entry, mel_tensor=mel_tensor)
    all_paths.append(absolute_path)
  return all_paths
