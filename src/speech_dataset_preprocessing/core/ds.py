import os
from collections import Counter
from dataclasses import dataclass
from logging import Logger
from typing import Callable, List, Optional, Set, Tuple

from speech_dataset_parser import (download_arctic, download_libritts,
                                   download_ljs, download_mailabs,
                                   download_thchs, download_thchs_kaldi,
                                   parse_arctic, parse_custom, parse_libritts,
                                   parse_ljs, parse_mailabs, parse_thchs,
                                   parse_thchs_kaldi)
from speech_dataset_parser.data import PreData, PreDataList
from speech_dataset_preprocessing.utils import (
    GenericList, remove_duplicates_list_orderpreserving)
from text_utils import (AccentsDict, Gender, IPAExtractionSettings, Language,
                        SpeakersDict, SpeakersLogDict, SymbolIdDict,
                        text_to_symbols)


@dataclass()
class DsData:
  entry_id: int
  basename: str
  speaker_name: str
  speaker_id: int
  text: str
  serialized_symbols: str
  serialized_accents: str
  wav_path: str
  lang: Language
  gender: Gender

  def load_init(self):
    self.lang = Language(self.lang)
    self.gender = Gender(self.gender)
    self.speaker_name = str(self.speaker_name)

  def __repr__(self):
    return str(self.entry_id)


class DsDataList(GenericList[DsData]):
  def load_init(self):
    for item in self.items():
      item.load_init()


def _preprocess_core(dir_path: str, auto_dl: bool, ignore_tones: Optional[bool], ignore_arcs: Optional[bool], replace_unknown_ipa_by: Optional[str], dl_func: Optional[Callable[[str], None]], parse_func: Callable[[str], PreDataList], logger: Logger) -> Tuple[
        SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict, AccentsDict]:
  if not os.path.isdir(dir_path) and auto_dl and dl_func is not None:
    dl_func(dir_path)

  data = parse_func(dir_path)
  speakers, speakers_log = _get_all_speakers(data)
  accents = _get_all_accents(data)
  ipa_settings = IPAExtractionSettings(
    ignore_tones=ignore_tones,
    ignore_arcs=ignore_arcs,
    replace_unknown_ipa_by=replace_unknown_ipa_by,
  )
  text_symbols = _extract_symbols(data, ipa_settings=ipa_settings, logger=logger)
  symbols = _get_symbols_id_dict(text_symbols)

  ds_data = DsDataList([get_dsdata_from_predata(
    values=x[0],
    text_symbols=x[1],
    i=i,
    speakers_dict=speakers,
    symbols=symbols,
    accents=accents
  ) for i, x in enumerate(zip(data.items(), text_symbols))])

  return speakers, speakers_log, symbols, accents, ds_data


def get_speaker_examples(data: DsDataList) -> DsDataList:
  processed_speakers: Set[str] = set()
  result = DsDataList()
  for values in data.items(True):
    if values.speaker_name not in processed_speakers:
      processed_speakers.add(values.speaker_name)
      result.append(values)
  return result


def thchs_preprocess(dir_path: str, auto_dl: bool, logger: Logger) -> Tuple[
        SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict, AccentsDict]:
  return _preprocess_core(
    dir_path=dir_path,
    auto_dl=auto_dl,
    ignore_tones=None,
    ignore_arcs=None,
    replace_unknown_ipa_by=None,
    dl_func=download_thchs,
    parse_func=parse_thchs,
    logger=logger,
  )


def custom_preprocess(dir_path: str, ignore_tones: Optional[bool], ignore_arcs: Optional[bool], replace_unknown_ipa_by: Optional[str], logger: Logger) -> Tuple[
  SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict, AccentsDict]:
  return _preprocess_core(
    dir_path=dir_path,
    auto_dl=False,
    ignore_tones=ignore_tones,
    ignore_arcs=ignore_arcs,
    replace_unknown_ipa_by=replace_unknown_ipa_by,
    dl_func=None,
    parse_func=parse_custom,
    logger=logger,
  )


def libritts_preprocess(dir_path: str, auto_dl: bool, logger: Logger) -> Tuple[
  SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict, AccentsDict]:
  return _preprocess_core(
      dir_path=dir_path,
      auto_dl=auto_dl,
      ignore_tones=None,
      ignore_arcs=None,
      replace_unknown_ipa_by=None,
      dl_func=download_libritts,
      parse_func=parse_libritts,
      logger=logger,
  )


def arctic_preprocess(dir_path: str, auto_dl: bool, logger: Logger) -> Tuple[
  SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict, AccentsDict]:
  return _preprocess_core(
    dir_path=dir_path,
    auto_dl=auto_dl,
    ignore_tones=None,
    ignore_arcs=None,
    replace_unknown_ipa_by=None,
    dl_func=download_arctic,
    parse_func=parse_arctic,
    logger=logger,
  )


def ljs_preprocess(dir_path: str, auto_dl: bool, logger: Logger) -> Tuple[
  SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict, AccentsDict]:
  return _preprocess_core(
    dir_path=dir_path,
    auto_dl=auto_dl,
    ignore_tones=None,
    ignore_arcs=None,
    replace_unknown_ipa_by=None,
    dl_func=download_ljs,
    parse_func=parse_ljs,
    logger=logger,
  )


def mailabs_preprocess(dir_path: str, auto_dl: bool, logger: Logger) -> Tuple[
  SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict, AccentsDict]:
  return _preprocess_core(
    dir_path=dir_path,
    auto_dl=auto_dl,
    ignore_tones=None,
    ignore_arcs=None,
    replace_unknown_ipa_by=None,
    dl_func=download_mailabs,
    parse_func=parse_mailabs,
    logger=logger,
  )


def thchs_kaldi_preprocess(dir_path: str, auto_dl: bool, logger: Logger) -> Tuple[SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict, AccentsDict]:
  return _preprocess_core(
    dir_path=dir_path,
    auto_dl=auto_dl,
    ignore_tones=None,
    ignore_arcs=None,
    replace_unknown_ipa_by=None,
    dl_func=download_thchs_kaldi,
    parse_func=parse_thchs_kaldi,
    logger=logger,
  )


def _get_all_speakers(l: PreDataList) -> Tuple[SpeakersDict, SpeakersLogDict]:
  all_speakers: List[str] = [x.speaker_name for x in l.items()]
  all_speakers_count = Counter(all_speakers)
  speakers_log = SpeakersLogDict.fromcounter(all_speakers_count)
  all_speakers = remove_duplicates_list_orderpreserving(all_speakers)
  speakers_dict = SpeakersDict.fromlist(all_speakers)
  return speakers_dict, speakers_log


def _get_all_accents(l: PreDataList) -> AccentsDict:
  accents = {x.speaker_accent for x in l.items()}
  return AccentsDict.init_from_accents(accents)


def _extract_symbols(l: PreDataList, ipa_settings: Optional[IPAExtractionSettings], logger: Logger) -> List[List[str]]:
  res: List[List[str]] = []
  for x in l.items():
    text_symbols = text_to_symbols(
      text=x.text,
      lang=x.lang,
      ipa_settings=ipa_settings,
      logger=logger,
    )
    res.append(text_symbols)
  return res


def _get_symbols_id_dict(text_symbols: List[List[str]]) -> SymbolIdDict:
  res = set()
  for x in text_symbols:
    res = res.union(set(x))
  return SymbolIdDict.init_from_symbols(res)


def get_dsdata_from_predata(values: PreData, i: int, speakers_dict: SpeakersDict, symbols: SymbolIdDict, accents: AccentsDict, text_symbols: List[str]) -> DsData:
  text_accents = [values.speaker_accent] * len(text_symbols)
  res = DsData(
    entry_id=i,
    basename=values.name,
    speaker_name=values.speaker_name,
    speaker_id=speakers_dict[values.speaker_name],
    text=values.text,
    serialized_symbols=symbols.get_serialized_ids(text_symbols),
    serialized_accents=accents.get_serialized_ids(text_accents),
    wav_path=values.wav_path,
    # TODO: lang to lang conversion
    lang=values.lang,
    gender=values.gender
  )
  return res
