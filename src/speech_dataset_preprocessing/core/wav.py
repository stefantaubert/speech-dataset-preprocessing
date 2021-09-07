"""
calculate wav duration and sampling rate
"""

from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import pandas as pd
from audio_utils import (get_duration_s, normalize_file, remove_silence_file,
                         stereo_to_mono_file, upsample_file)
from audio_utils.mel import TacotronSTFT, TSTFTHParams
from numpy.core.fromnumeric import mean
from scipy.io.wavfile import read, write
from speech_dataset_preprocessing.core.ds import DsDataList
from speech_dataset_preprocessing.globals import DEFAULT_PRE_CHUNK_SIZE
from speech_dataset_preprocessing.utils import GenericList, get_chunk_name
from text_utils.types import Speaker


@dataclass()
class WavData:
  entry_id: int
  relative_wav_path: Path
  duration: float
  sr: int
  #size: float
  #is_stereo: bool

  def __repr__(self):
    return str(self.entry_id)


class WavDataList(GenericList[WavData]):
  def get_entry(self, entry_id: int) -> WavData:
    for entry in self.items():
      if entry.entry_id == entry_id:
        return entry
    raise Exception(f"Entry {entry_id} not found.")


def log_stats(ds_data: DsDataList, wav_data: WavDataList):
  logger = getLogger(__name__)
  if len(wav_data) > 0:
    logger.info(f"Sampling rate: {wav_data.items()[0].sr}")
  stats: List[str, int, float, float, float, int] = []

  durations = [x.duration for x in wav_data.items()]
  stats.append((
    "Overall",
    len(wav_data),
    min(durations),
    max(durations),
    mean(durations),
    sum(durations) / 60,
    sum(durations) / 3600,
  ))
  speaker_durations: Dict[Speaker, List[float]] = {}
  for ds_entry, wav_entry in zip(ds_data.items(), wav_data.items()):
    if ds_entry.speaker_name not in speaker_durations:
      speaker_durations[ds_entry.speaker_name] = []
    speaker_durations[ds_entry.speaker_name].append(wav_entry.duration)
  for speaker_name, speaker_durations in speaker_durations.items():
    stats.append((
      speaker_name,
      len(speaker_durations),
      min(speaker_durations),
      max(speaker_durations),
      mean(speaker_durations),
      sum(speaker_durations) / 60,
      sum(speaker_durations) / 3600,
    ))

  stats.sort(key=lambda x: (x[-2]), reverse=True)
  stats_csv = pd.DataFrame(stats, columns=[
    "Speaker",
    "# Entries",
    "Min (s)",
    "Max (s)",
    "Avg (s)",
    "Total (min)",
    "Total (h)",
  ])

  with pd.option_context(
    'display.max_rows', None,
    'display.max_columns', None,
    'display.width', None,
    'display.precision', 4,
  ):
    logger.info(stats_csv)


def preprocess(data: DsDataList, dest_dir: Path) -> WavDataList:
  result = WavDataList()

  for values in data.items(True):
    sampling_rate, wav = read(values.wav_path)
    duration = get_duration_s(wav, sampling_rate)

    chunk_dir_name = get_chunk_name(
      i=values.entry_id,
      chunksize=DEFAULT_PRE_CHUNK_SIZE,
      maximum=len(data) - 1
    )
    absolute_chunk_dir = dest_dir / chunk_dir_name
    absolute_chunk_dir.mkdir(parents=True, exist_ok=True)
    relative_dest_wav_path = Path(chunk_dir_name) / f"{values!r}.wav"
    absolute_dest_wav_path = dest_dir / relative_dest_wav_path
    write(absolute_dest_wav_path, sampling_rate, wav)

    wav_data = WavData(values.entry_id, relative_dest_wav_path, duration, sampling_rate)
    result.append(wav_data)

  return result


def resample(data: WavDataList, orig_dir: Path, dest_dir: Path, new_rate: int) -> WavDataList:
  result = WavDataList()

  for values in data.items(True):
    chunk_dir_name = get_chunk_name(
      i=values.entry_id,
      chunksize=DEFAULT_PRE_CHUNK_SIZE,
      maximum=len(data) - 1
    )
    absolute_chunk_dir = dest_dir / chunk_dir_name
    absolute_chunk_dir.mkdir(parents=True, exist_ok=True)
    relative_dest_wav_path = Path(chunk_dir_name) / f"{values!r}.wav"
    absolute_dest_wav_path = dest_dir / relative_dest_wav_path

    # TODO assert not is_overamp
    absolute_orig_wav_path = orig_dir / values.relative_wav_path
    upsample_file(absolute_orig_wav_path, absolute_dest_wav_path, new_rate)
    wav_data = WavData(values.entry_id, relative_dest_wav_path, values.duration, new_rate)
    result.append(wav_data)

  return result


def stereo_to_mono(data: WavDataList, orig_dir: Path, dest_dir: Path) -> WavDataList:
  result = WavDataList()

  for values in data.items(True):
    chunk_dir_name = get_chunk_name(
      i=values.entry_id,
      chunksize=DEFAULT_PRE_CHUNK_SIZE,
      maximum=len(data) - 1
    )
    absolute_chunk_dir = dest_dir / chunk_dir_name
    absolute_chunk_dir.mkdir(parents=True, exist_ok=True)
    relative_dest_wav_path = Path(chunk_dir_name) / f"{values!r}.wav"
    absolute_dest_wav_path = dest_dir / relative_dest_wav_path

    # todo assert not is_overamp
    absolute_orig_wav_path = orig_dir / values.relative_wav_path
    stereo_to_mono_file(absolute_orig_wav_path, absolute_dest_wav_path)

    wav_data = WavData(values.entry_id, relative_dest_wav_path, values.duration, values.sr)
    result.append(wav_data)

  return result


def remove_silence(data: WavDataList, orig_dir: Path, dest_dir: Path, chunk_size: int, threshold_start: float, threshold_end: float, buffer_start_ms: float, buffer_end_ms: float) -> WavDataList:
  result = WavDataList()

  for values in data.items(True):
    chunk_dir_name = get_chunk_name(
      i=values.entry_id,
      chunksize=DEFAULT_PRE_CHUNK_SIZE,
      maximum=len(data) - 1
    )
    absolute_chunk_dir = dest_dir / chunk_dir_name
    absolute_chunk_dir.mkdir(parents=True, exist_ok=True)
    relative_dest_wav_path = Path(chunk_dir_name) / f"{values!r}.wav"
    absolute_dest_wav_path = dest_dir / relative_dest_wav_path

    absolute_orig_wav_path = orig_dir / values.relative_wav_path
    new_duration = remove_silence_file(
      in_path=absolute_orig_wav_path,
      out_path=absolute_dest_wav_path,
      chunk_size=chunk_size,
      threshold_start=threshold_start,
      threshold_end=threshold_end,
      buffer_start_ms=buffer_start_ms,
      buffer_end_ms=buffer_end_ms
    )

    wav_data = WavData(values.entry_id, relative_dest_wav_path, new_duration, values.sr)
    result.append(wav_data)

  return result


def remove_silence_plot(wav_path: Path, out_path: Path, chunk_size: int, threshold_start: float, threshold_end: float, buffer_start_ms: float, buffer_end_ms: float):
  remove_silence_file(
    in_path=wav_path,
    out_path=out_path,
    chunk_size=chunk_size,
    threshold_start=threshold_start,
    threshold_end=threshold_end,
    buffer_start_ms=buffer_start_ms,
    buffer_end_ms=buffer_end_ms
  )

  sampling_rate, _ = read(wav_path)

  hparams = TSTFTHParams()
  hparams.sampling_rate = sampling_rate
  plotter = TacotronSTFT(hparams, logger=getLogger())

  mel_orig = plotter.get_mel_tensor_from_file(wav_path)
  mel_trimmed = plotter.get_mel_tensor_from_file(out_path)

  return mel_orig, mel_trimmed


def normalize(data: WavDataList, orig_dir: Path, dest_dir: Path) -> WavDataList:
  result = WavDataList()

  for values in data.items(True):
    chunk_dir_name = get_chunk_name(
      i=values.entry_id,
      chunksize=DEFAULT_PRE_CHUNK_SIZE,
      maximum=len(data) - 1
    )
    absolute_chunk_dir = dest_dir / chunk_dir_name
    absolute_chunk_dir.mkdir(parents=True, exist_ok=True)
    relative_dest_wav_path = Path(chunk_dir_name) / f"{values!r}.wav"
    absolute_dest_wav_path = dest_dir / relative_dest_wav_path

    absolute_orig_wav_path = orig_dir / values.relative_wav_path
    normalize_file(absolute_orig_wav_path, absolute_dest_wav_path)

    wav_data = WavData(values.entry_id, relative_dest_wav_path, values.duration, values.sr)
    result.append(wav_data)

  return result
