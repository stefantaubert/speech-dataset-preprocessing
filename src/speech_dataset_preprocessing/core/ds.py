from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Set, Tuple

from speech_dataset_parser import (download_ljs, download_thchs,
                                   download_thchs_kaldi, parse_arctic,
                                   parse_custom, parse_libritts, parse_ljs,
                                   parse_mailabs, parse_thchs,
                                   parse_thchs_kaldi)
from speech_dataset_parser.data import PreData, PreDataList
from speech_dataset_preprocessing.utils import GenericList
from text_utils import (Gender, Language, Speaker, Speakers, SpeakersLogDict,
                        SymbolFormat, Symbols, get_format_from_str,
                        get_lang_from_str, text_to_symbols)


@dataclass()
class DsData:
  entry_id: int
  identifier: str
  symbols: Symbols
  symbols_format: SymbolFormat
  symbols_language: Language
  speaker_name: Speaker
  speaker_gender: Gender
  wav_absolute_path: Path

  def __repr__(self):
    return str(self.entry_id)


class DsDataList(GenericList[DsData]):
  pass


PreprocessingResult = Tuple[SpeakersLogDict, DsDataList]


def _preprocess_core(dir_path: Path, auto_dl: bool, dl_func: Optional[Callable[[str], None]], parse_func: Callable[[Path], PreDataList]) -> PreprocessingResult:
  if not dir_path.is_dir() and auto_dl and dl_func is not None:
    dl_func(dir_path)

  data = parse_func(dir_path)
  speakers_log = _get_all_speakers(data)

  ds_data = DsDataList([get_dsdata_from_predata(predata, i)
                        for i, predata in enumerate(data.items())])

  return speakers_log, ds_data


def get_dsdata_from_predata(predata: PreData, i: int) -> DsData:
  text_language = get_lang_from_str(str(predata.text_language))
  text_format = get_format_from_str(str(predata.text_format))
  res = DsData(
    entry_id=i,
    identifier=predata.identifier,
    symbols=text_to_symbols(
      text=predata.text,
      lang=text_language,
      text_format=text_format,
    ),
    symbols_format=text_format,
    symbols_language=text_language,
    wav_absolute_path=predata.wav_path,
    speaker_name=predata.speaker_name,
    speaker_gender=predata.speaker_gender,
  )
  return res


def get_speaker_examples(data: DsDataList) -> DsDataList:
  processed_speakers: Set[str] = set()
  result = DsDataList()
  for values in data.items(True):
    if values.speaker_name not in processed_speakers:
      processed_speakers.add(values.speaker_name)
      result.append(values)
  return result


def thchs_preprocess(dir_path: Path, auto_dl: bool) -> PreprocessingResult:
  return _preprocess_core(
    dir_path=dir_path,
    auto_dl=auto_dl,
    dl_func=download_thchs,
    parse_func=parse_thchs,
  )


def custom_preprocess(dir_path: Path) -> PreprocessingResult:
  return _preprocess_core(
    dir_path=dir_path,
    auto_dl=False,
    dl_func=None,
    parse_func=parse_custom,
  )


def libritts_preprocess(dir_path: Path) -> PreprocessingResult:
  return _preprocess_core(
      dir_path=dir_path,
      auto_dl=False,
      dl_func=None,
      parse_func=parse_libritts,
  )


def arctic_preprocess(dir_path: Path) -> PreprocessingResult:
  return _preprocess_core(
    dir_path=dir_path,
    auto_dl=False,
    dl_func=None,
    parse_func=parse_arctic,
  )


def ljs_preprocess(dir_path: Path, auto_dl: bool) -> PreprocessingResult:
  return _preprocess_core(
    dir_path=dir_path,
    auto_dl=auto_dl,
    dl_func=download_ljs,
    parse_func=parse_ljs,
  )


def mailabs_preprocess(dir_path: Path) -> PreprocessingResult:
  return _preprocess_core(
    dir_path=dir_path,
    auto_dl=False,
    dl_func=None,
    parse_func=parse_mailabs,
  )


def thchs_kaldi_preprocess(dir_path: Path, auto_dl: bool) -> PreprocessingResult:
  return _preprocess_core(
    dir_path=dir_path,
    auto_dl=auto_dl,
    dl_func=download_thchs_kaldi,
    parse_func=parse_thchs_kaldi,
  )


def _get_all_speakers(l: PreDataList) -> SpeakersLogDict:
  all_speakers: Speakers = [x.speaker_name for x in l.items()]
  all_speakers_count = Counter(all_speakers)
  speakers_log = SpeakersLogDict.fromcounter(all_speakers_count)
  return speakers_log
