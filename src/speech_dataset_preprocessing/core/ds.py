import os
from collections import Counter
from dataclasses import dataclass
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
from text_utils import (Gender, Language, Speakers, SpeakersDict,
                        SpeakersLogDict, SymbolIdDict, Symbols,
                        text_to_symbols)


@dataclass()
class DsData:
  entry_id: int
  basename: str
  speaker_name: str
  speaker_id: int
  text: str
  serialized_symbols: str
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


def _preprocess_core(dir_path: str, auto_dl: bool, dl_func: Optional[Callable[[str], None]], parse_func: Callable[[str], PreDataList]) -> Tuple[SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict]:
  if not os.path.isdir(dir_path) and auto_dl and dl_func is not None:
    dl_func(dir_path)

  data = parse_func(dir_path)
  speakers, speakers_log = _get_all_speakers(data)
  text_symbols = _extract_symbols(data)
  symbols = _get_symbols_id_dict(text_symbols)

  ds_data = DsDataList([get_dsdata_from_predata(
    values=x[0],
    text_symbols=x[1],
    i=i,
    speakers_dict=speakers,
    symbols=symbols,
  ) for i, x in enumerate(zip(data.items(), text_symbols))])

  return speakers, speakers_log, symbols, ds_data


def get_speaker_examples(data: DsDataList) -> DsDataList:
  processed_speakers: Set[str] = set()
  result = DsDataList()
  for values in data.items(True):
    if values.speaker_name not in processed_speakers:
      processed_speakers.add(values.speaker_name)
      result.append(values)
  return result


def thchs_preprocess(dir_path: str, auto_dl: bool) -> Tuple[
        SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict]:
  return _preprocess_core(
    dir_path=dir_path,
    auto_dl=auto_dl,
    dl_func=download_thchs,
    parse_func=parse_thchs,
  )


def custom_preprocess(dir_path: str) -> Tuple[
  SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict]:
  return _preprocess_core(
    dir_path=dir_path,
    auto_dl=False,
    dl_func=None,
    parse_func=parse_custom,
  )


def libritts_preprocess(dir_path: str, auto_dl: bool) -> Tuple[
  SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict]:
  return _preprocess_core(
      dir_path=dir_path,
      auto_dl=auto_dl,
      dl_func=download_libritts,
      parse_func=parse_libritts,
  )


def arctic_preprocess(dir_path: str, auto_dl: bool) -> Tuple[
  SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict]:
  return _preprocess_core(
    dir_path=dir_path,
    auto_dl=auto_dl,
    dl_func=download_arctic,
    parse_func=parse_arctic,
  )


def ljs_preprocess(dir_path: str, auto_dl: bool) -> Tuple[
  SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict]:
  return _preprocess_core(
    dir_path=dir_path,
    auto_dl=auto_dl,
    dl_func=download_ljs,
    parse_func=parse_ljs,
  )


def mailabs_preprocess(dir_path: str, auto_dl: bool) -> Tuple[
  SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict]:
  return _preprocess_core(
    dir_path=dir_path,
    auto_dl=auto_dl,
    dl_func=download_mailabs,
    parse_func=parse_mailabs,
  )


def thchs_kaldi_preprocess(dir_path: str, auto_dl: bool) -> Tuple[SpeakersDict, SpeakersLogDict, DsDataList, SymbolIdDict]:
  return _preprocess_core(
    dir_path=dir_path,
    auto_dl=auto_dl,
    dl_func=download_thchs_kaldi,
    parse_func=parse_thchs_kaldi,
  )


def _get_all_speakers(l: PreDataList) -> Tuple[SpeakersDict, SpeakersLogDict]:
  all_speakers: Speakers = [x.speaker_name for x in l.items()]
  all_speakers_count = Counter(all_speakers)
  speakers_log = SpeakersLogDict.fromcounter(all_speakers_count)
  all_speakers = remove_duplicates_list_orderpreserving(all_speakers)
  speakers_dict = SpeakersDict.fromlist(all_speakers)
  return speakers_dict, speakers_log


def _extract_symbols(pre_data: PreDataList) -> List[Symbols]:
  res: List[Symbols] = []
  for pre_data_entry in pre_data.items():
    text_symbols = text_to_symbols(
      text=pre_data_entry.text,
      lang=pre_data_entry.lang,
    )
    res.append(text_symbols)
  return res


def _get_symbols_id_dict(text_symbols: List[Symbols]) -> SymbolIdDict:
  all_symbols = {symbol for symbols in text_symbols for symbol in symbols}
  return SymbolIdDict.init_from_symbols(all_symbols)


def get_dsdata_from_predata(values: PreData, i: int, speakers_dict: SpeakersDict, symbols: SymbolIdDict, text_symbols: Symbols) -> DsData:
  res = DsData(
    entry_id=i,
    basename=values.name,
    speaker_name=values.speaker_name,
    speaker_id=speakers_dict[values.speaker_name],
    text=values.text,
    serialized_symbols=symbols.get_serialized_ids(text_symbols),
    wav_path=values.wav_path,
    # TODO: lang to lang conversion
    lang=values.lang,
    gender=values.gender
  )
  return res
