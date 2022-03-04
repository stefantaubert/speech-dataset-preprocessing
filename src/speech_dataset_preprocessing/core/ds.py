from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Set, Tuple

from general_utils import GenericList
from speech_dataset_parser_api import parse_directory
from speech_dataset_parser_old import (PreData, PreDataList, download_ljs,
                                       download_thchs, download_thchs_kaldi,
                                       parse_arctic, parse_libritts, parse_ljs,
                                       parse_mailabs, parse_thchs,
                                       parse_thchs_kaldi)
from text_utils import (Gender, Language, Speaker, Speakers, SpeakersLogDict,
                        SymbolFormat, Symbols, get_format_from_str,
                        get_lang_from_str)


@dataclass()
class DsData:
  entry_id: int
  basename: str
  symbols: Symbols
  symbols_format: SymbolFormat
  symbols_language: Language
  speaker_name: Speaker
  speaker_gender: Optional[Gender]
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
  data_identifiers_are_unique = len(set(x.identifier for x in data.items())) == len(data)
  assert data_identifiers_are_unique

  speakers_log = _get_all_speakers(data)

  ds_data = DsDataList([get_dsdata_from_predata(dir_path, predata)
                        for predata in data.items()])
  for entry in ds_data.items():
    assert entry.wav_absolute_path.is_file()

  return speakers_log, ds_data


def generic_preprocess(directory: Path, tier_name: str, n_digits: int, symbols_format: SymbolFormat) -> PreprocessingResult:
  result = DsDataList()
  entries = parse_directory(directory, tier_name, n_digits)
  for entry_nr, entry in enumerate(entries):
    data_entry = DsData(
      entry_id=entry_nr,
      basename=entry.audio_file_rel.stem,
      symbols=entry.symbols,
      symbols_format=symbols_format,
      symbols_language=get_lang_from_iso(entry.symbols_language),
      wav_absolute_path=directory / entry.audio_file_rel,
      speaker_name=entry.speaker_name,
      speaker_gender=get_gender_from_iso(entry.speaker_gender),
    )
    assert data_entry.wav_absolute_path.is_file()
    result.append(data_entry)

  all_speakers: Speakers = (x.speaker_name for x in result.items())
  all_speakers_counter = Counter(all_speakers)
  speakers_log = SpeakersLogDict.fromcounter(all_speakers_counter)

  return speakers_log, result


def get_gender_from_iso(gender_iso: int) -> Optional[Gender]:
  if gender_iso in {0, 9}:
    return None
  if gender_iso == 1:
    return Gender.MALE
  if gender_iso == 2:
    return Gender.FEMALE
  assert False


def get_lang_from_iso(lang_iso: str) -> Language:
  if lang_iso == "eng":
    return Language.ENG
  if lang_iso == "deu":
    return Language.GER
  if lang_iso == "zho":
    return Language.CHN
  assert False


def get_dsdata_from_predata(dir_path: Path, predata: PreData) -> DsData:
  text_language = get_lang_from_str(str(predata.symbols_language))
  text_format = get_format_from_str(str(predata.symbols_format))
  res = DsData(
    entry_id=predata.identifier,
    basename=predata.basename,
    symbols=predata.symbols,
    symbols_format=text_format,
    symbols_language=text_language,
    wav_absolute_path=dir_path / predata.relative_audio_path,
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
