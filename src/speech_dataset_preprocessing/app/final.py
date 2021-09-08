from logging import getLogger
from pathlib import Path

from speech_dataset_preprocessing.app.ds import get_ds_dir, load_ds_data
from speech_dataset_preprocessing.app.mel import get_mel_dir, load_mel_data
from speech_dataset_preprocessing.app.text import get_text_dir, load_text_data
from speech_dataset_preprocessing.app.wav import get_wav_dir, load_wav_data
from speech_dataset_preprocessing.core.final import (FinalDsEntryList,
                                                     get_final_ds_from_data)


def get_final_ds(base_dir: Path, ds_name: str, text_name: str, wav_name: str) -> FinalDsEntryList:
  logger = getLogger(__name__)
  ds_dir = get_ds_dir(base_dir, ds_name, create=False)
  if not ds_dir.is_dir() or not ds_dir.exists():
    logger.info("Dataset not found.")
    return

  text_dir = get_text_dir(ds_dir, text_name)
  if not text_dir.is_dir() or not text_dir.exists():
    logger.info("Text data not found.")
    return

  wav_dir = get_wav_dir(ds_dir, wav_name)
  if not wav_dir.is_dir() or not wav_dir.exists():
    logger.info("Wav data not found.")
    return

  mel_dir = get_mel_dir(ds_dir, wav_name)
  if not mel_dir.is_dir() or not mel_dir.exists():
    logger.info("Wav data not found.")
    return

  ds_data = load_ds_data(ds_dir)
  text_data = load_text_data(text_dir)
  wav_data = load_wav_data(wav_dir)
  mel_data = load_mel_data(mel_dir)

  result = get_final_ds_from_data(
    ds_data=ds_data,
    text_data=text_data,
    wav_data=wav_data,
    mel_data=mel_data,
    wav_dir=wav_dir,
    mel_dir=mel_dir,
  )

  return result
