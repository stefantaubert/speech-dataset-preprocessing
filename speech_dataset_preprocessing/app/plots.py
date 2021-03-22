import os
from functools import partial
from tempfile import mktemp
from typing import Dict, Optional

from audio_utils.mel import plot_melspec
from image_utils import stack_images_horizontally, stack_images_vertically
from matplotlib import pyplot as plt
from speech_dataset_preprocessing.app.ds import get_ds_dir, load_ds_csv
from speech_dataset_preprocessing.app.wav import get_wav_dir, load_wav_csv
from speech_dataset_preprocessing.core.ds import DsData
from speech_dataset_preprocessing.core.plots import process
from speech_dataset_preprocessing.core.wav import WavData
from speech_dataset_preprocessing.globals import DEFAULT_PRE_CHUNK_SIZE
from speech_dataset_preprocessing.utils import (get_chunk_name, get_filepaths,
                                                get_subdir, get_subfolders,
                                                make_batches_h_v)
from torch import Tensor
from tqdm import tqdm

VERTICAL_COUNT = 10
HORIZONTAL_COUNT = 4


def _get_plots_root_dir(ds_dir: str, create: bool = False):
  return get_subdir(ds_dir, "plots", create)


def get_plots_dir(ds_dir: str, mel_name: str, create: bool = False):
  return get_subdir(_get_plots_root_dir(ds_dir, create), mel_name, create)


def save_plot(dest_dir: str, data_len: int, wav_entry: WavData, ds_entry: DsData, mel_tensor: Tensor) -> str:
  chunk_dir = os.path.join(dest_dir, get_chunk_name(
    wav_entry.entry_id, chunksize=DEFAULT_PRE_CHUNK_SIZE, maximum=data_len - 1))
  os.makedirs(chunk_dir, exist_ok=True)

  plot_melspec(mel_tensor, title=f"{repr(wav_entry)}: {ds_entry.text}")
  absolute_path = os.path.join(chunk_dir, f"{repr(wav_entry)}.png")
  plt.savefig(absolute_path, bbox_inches='tight')
  plt.close()

  return absolute_path


def plot_mels(base_dir: str, ds_name: str, wav_name: str, custom_hparams: Optional[Dict[str, str]] = None):
  print("Plotting wav mel spectograms...")
  ds_dir = get_ds_dir(base_dir, ds_name)
  plots_dir = get_plots_dir(ds_dir, wav_name)
  if os.path.isdir(plots_dir):
    print("Already exists.")
  else:
    wav_dir = get_wav_dir(ds_dir, wav_name)
    assert os.path.isdir(wav_dir)
    data = load_wav_csv(wav_dir)
    ds_data = load_ds_csv(ds_dir)
    assert len(data) > 0
    save_callback = partial(save_plot, dest_dir=plots_dir, data_len=len(data))
    all_absolute_paths = process(data, ds_data, wav_dir, custom_hparams, save_callback)

    # all_paths = get_all_paths(plots_dir)

    batches = make_batches_h_v(all_absolute_paths, VERTICAL_COUNT, HORIZONTAL_COUNT)

    plot_batches_h_v(batches, plots_dir)


def get_all_paths(plots_dir):
  all_subs = get_subfolders(plots_dir)
  all_paths = []
  for sub in all_subs:
    paths = get_filepaths(sub)
    all_paths.extend(paths)
  return all_paths


def plot_batches_v_h(batches, plots_dir):
  for i, h_batch in enumerate(tqdm(batches)):
    v_files = []
    for v_batch in h_batch:
      v_path = mktemp(suffix=".png")
      stack_images_vertically(v_batch, v_path)
      v_files.append(v_path)
    outpath = os.path.join(plots_dir, f"{i}.png")
    stack_images_horizontally(v_files, outpath)
    for v_file in v_files:
      os.remove(v_file)


def plot_batches_h_v(batches, plots_dir):
  for i, v_batch in enumerate(tqdm(batches)):
    h_files = []
    for h_batch in v_batch:
      h_path = mktemp(suffix=".png")
      stack_images_horizontally(h_batch, h_path)
      h_files.append(h_path)
    outpath = os.path.join(plots_dir, f"{i}.png")
    stack_images_vertically(h_files, outpath)
    for v_file in h_files:
      os.remove(v_file)
