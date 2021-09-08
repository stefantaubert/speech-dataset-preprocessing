import os
import tempfile
from logging import getLogger
from pathlib import Path
from shutil import copyfile
from typing import List, Optional

import matplotlib.pylab as plt
import numpy as np
from audio_utils.mel import plot_melspec
from image_utils import stack_images_vertically
from speech_dataset_preprocessing.app.ds import get_ds_dir
from speech_dataset_preprocessing.app.wav import get_wav_dir, load_wav_data
from speech_dataset_preprocessing.core.wav import WavData
from speech_dataset_preprocessing.core.wav import \
    remove_silence_plot as remove_silence_plot_core
from speech_dataset_preprocessing.utils import get_subdir


def _save_orig_plot_if_not_exists(dest_dir: Path, mel) -> Path:
  path = dest_dir / "original.png"
  if not path.is_file():
    plot_melspec(mel, title="Original")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
  return path


def _save_orig_wav_if_not_exists(dest_dir: Path, orig_path: Path) -> None:
  path = dest_dir / "original.wav"
  if not path.is_file():
    copyfile(orig_path, path)


def _save_trimmed_plot_temp(mel: np.ndarray) -> Path:
  path = Path(tempfile.mktemp(suffix=".png"))
  plot_melspec(mel, title="Trimmed")
  plt.savefig(path, bbox_inches='tight')
  plt.close()
  return path


def _save_comparison(dest_dir: Path, dest_name: str, paths: List[Path]) -> str:
  path = dest_dir / f"{dest_name}.png"
  stack_images_vertically(paths, path)
  return path


def _get_trim_root_dir(wav_dir: Path) -> Path:
  return get_subdir(wav_dir, "trim", create=True)


def _get_trim_dir(wav_dir: Path, entry: WavData) -> Path:
  return _get_trim_root_dir(wav_dir) / str(entry.entry_id)


def remove_silence_plot(base_dir: Path, ds_name: str, wav_name: str, chunk_size: int, threshold_start: float, threshold_end: float, buffer_start_ms: float, buffer_end_ms: float, entry_id: Optional[int] = None) -> None:
  ds_dir = get_ds_dir(base_dir, ds_name)
  wav_dir = get_wav_dir(ds_dir, wav_name)
  assert os.path.isdir(wav_dir)
  data = load_wav_data(wav_dir)
  if entry_id is None:
    entry = data.get_random_entry()
  else:
    entry = data.get_entry(entry_id)

  dest_dir = _get_trim_dir(wav_dir, entry)
  dest_dir.mkdir(parents=True, exist_ok=True)

  dest_name = f"cs={chunk_size},ts={threshold_start}dBFS,bs={buffer_start_ms}ms,te={threshold_end}dBFS,be={buffer_end_ms}ms"

  wav_trimmed = dest_dir / f"{dest_name}.wav"
  absolute_wav_path = wav_dir / entry.wav_relative_path

  mel_orig, mel_trimmed = remove_silence_plot_core(
    wav_path=absolute_wav_path,
    out_path=wav_trimmed,
    chunk_size=chunk_size,
    threshold_start=threshold_start,
    threshold_end=threshold_end,
    buffer_start_ms=buffer_start_ms,
    buffer_end_ms=buffer_end_ms
  )

  _save_orig_wav_if_not_exists(dest_dir, absolute_wav_path)
  orig = _save_orig_plot_if_not_exists(dest_dir, mel_orig)
  trimmed = _save_trimmed_plot_temp(mel_trimmed)
  resulting_path = _save_comparison(dest_dir, dest_name, [orig, trimmed])
  os.remove(trimmed)
  logger = getLogger(__name__)
  logger.info(f"Saved result to: {resulting_path}")
