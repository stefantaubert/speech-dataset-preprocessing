from dataclasses import dataclass
from pathlib import Path

from pandas import DataFrame
from speech_dataset_preprocessing.core.ds import DsDataList
from speech_dataset_preprocessing.core.mel import MelDataList
from speech_dataset_preprocessing.core.text import TextDataList
from speech_dataset_preprocessing.core.wav import WavDataList
from speech_dataset_preprocessing.utils import GenericList
from text_utils import Gender, Language, Speaker, SymbolFormat, Symbols


@dataclass
class FinalDsEntry():
  entry_id: int
  identifier: str
  speaker_name: Speaker
  speaker_gender: Gender
  symbols_language: Language
  symbols_original: Symbols
  symbols_original_format: SymbolFormat
  symbols: Symbols
  symbols_format: SymbolFormat
  wav_original_absolute_path: Path
  wav_absolute_path: Path
  wav_duration: float
  wav_sampling_rate: int
  mel_absolute_path: Path
  mel_n_channels: int


class FinalDsEntryList(GenericList[FinalDsEntry]):
  pass


def get_analysis_df(data: FinalDsEntryList) -> DataFrame:
  values = [
    (
      entry.identifier,
      entry.speaker_name,
      repr(entry.symbols_language),
      ''.join(entry.symbols_original),
      repr(entry.symbols_original_format),
      ''.join(entry.symbols),
      repr(entry.symbols_format),
      str(entry.wav_original_absolute_path),
      str(entry.wav_absolute_path),
      entry.wav_duration,
      entry.wav_sampling_rate,
      str(entry.mel_absolute_path),
      entry.mel_n_channels,
    ) for entry in data.items()
  ]

  columns = [
    "Id",
    "Speaker",
    "Language",
    "Original symbols",
    "Original symbols format",
    "Symbols",
    "Symbols format",
    "Original wav-path",
    "Wav-path",
    "Wav duration (s)",
    "Wav sampling rate (Hz)",
    "Mel-path",
    "# Mel-channels",
  ]

  result = DataFrame(data=values, columns=columns)
  return result


def get_final_ds_from_data(ds_data: DsDataList, text_data: TextDataList, wav_data: WavDataList, mel_data: MelDataList, wav_dir: Path, mel_dir: Path) -> FinalDsEntryList:
  res = WavDataList()
  for ds_data_entry, text_data_entry, wav_data_entry, mel_data_entry in zip(ds_data.items(), text_data.items(), wav_data.items(), mel_data.items()):
    assert ds_data_entry.entry_id == text_data_entry.entry_id == wav_data_entry.entry_id == mel_data_entry.entry_id
    assert ds_data_entry.symbols_language == text_data_entry.symbols_language

    new_entry = FinalDsEntry(
      entry_id=ds_data_entry.entry_id,
      identifier=ds_data_entry.identifier,
      speaker_gender=ds_data_entry.speaker_gender,
      speaker_name=ds_data_entry.speaker_name,
      symbols_language=ds_data_entry.symbols_language,
      symbols_original=ds_data_entry.symbols,
      symbols_original_format=ds_data_entry.symbols_format,
      symbols=text_data_entry.symbols,
      symbols_format=text_data_entry.symbols_format,
      wav_original_absolute_path=ds_data_entry.wav_absolute_path,
      wav_absolute_path=wav_dir / wav_data_entry.wav_relative_path,
      wav_duration=wav_data_entry.wav_duration,
      wav_sampling_rate=wav_data_entry.wav_sampling_rate,
      mel_absolute_path=mel_dir / mel_data_entry.mel_relative_path,
      mel_n_channels=mel_data_entry.mel_n_channels,
    )

    res.append(new_entry)

  return res
