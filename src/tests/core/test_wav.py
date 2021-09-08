from pathlib import Path

from speech_dataset_preprocessing.core.wav import WavData


def test_repr():
  result = WavData(5, Path(""), 0.0, 0)
  assert repr(result) == "5"
