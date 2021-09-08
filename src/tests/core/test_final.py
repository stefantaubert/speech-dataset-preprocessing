from pathlib import Path

from speech_dataset_preprocessing.core.ds import DsData, DsDataList
from speech_dataset_preprocessing.core.final import get_final_ds_from_data
from speech_dataset_preprocessing.core.mel import MelData, MelDataList
from speech_dataset_preprocessing.core.text import TextData, TextDataList
from speech_dataset_preprocessing.core.wav import WavData, WavDataList
from text_utils.gender import Gender
from text_utils.language import Language
from text_utils.symbol_format import SymbolFormat


def test_get_final_ds_from_data():
  ds_data = DsData(
    entry_id=1,
    identifier="identifier",
    speaker_gender=Gender.FEMALE,
    speaker_name="Speaker 1",
    symbols=("a", "b",),
    symbols_format=SymbolFormat.GRAPHEMES,
    symbols_language=Language.ENG,
    wav_absolute_path=Path("test.wav"),
  )

  text_data = TextData(
    entry_id=ds_data.entry_id,
    symbols=("ʊ", "ɪ",),
    symbols_format=SymbolFormat.PHONEMES_IPA,
    symbols_language=ds_data.symbols_language,
  )

  wav_data = WavData(
    entry_id=ds_data.entry_id,
    wav_duration=7.5,
    wav_relative_path=Path("test2.wav"),
    wav_sampling_rate=22050,
  )

  mel_data = MelData(
    entry_id=ds_data.entry_id,
    mel_n_channels=80,
    mel_relative_path=Path("test2.pt"),
  )

  result = get_final_ds_from_data(
    ds_data=DsDataList([ds_data]),
    mel_data=MelDataList([mel_data]),
    mel_dir=Path("meldir"),
    text_data=TextDataList([text_data]),
    wav_data=WavDataList([wav_data]),
    wav_dir=Path("wavdir"),
  )

  assert len(result) == 1
  result_first_entry = result.items()[0]
  assert result_first_entry.entry_id == 1
  assert result_first_entry.identifier == "identifier"
  assert result_first_entry.speaker_gender == Gender.FEMALE
  assert result_first_entry.speaker_name == "Speaker 1"
  assert result_first_entry.symbols_format == SymbolFormat.PHONEMES_IPA
  assert result_first_entry.symbols == ("ʊ", "ɪ",)
  assert result_first_entry.symbols_original == ("a", "b",)
  assert result_first_entry.symbols_original_format == SymbolFormat.GRAPHEMES
  assert result_first_entry.wav_duration == 7.5
  assert result_first_entry.wav_sampling_rate == 22050
  assert result_first_entry.wav_absolute_path == Path("wavdir/test2.wav")
  assert result_first_entry.wav_original_absolute_path == Path("test.wav")
  assert result_first_entry.mel_absolute_path == Path("meldir/test2.pt")
  assert result_first_entry.mel_n_channels == 80
