from collections import Counter
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from logging import getLogger
from typing import Dict, List, Optional

import pandas as pd
from general_utils import GenericList
from numpy.core.fromnumeric import mean
from sentence2pronunciation.lookup_cache import LookupCache, get_empty_cache
from speech_dataset_preprocessing.core.ds import DsDataList
from text_utils import EngToIPAMode, Language, Speaker, SymbolFormat, Symbols
from text_utils import change_ipa as change_ipa_method
from text_utils import symbols_to_ipa, text_normalize, text_to_symbols
from text_utils.pronunciation.main import prepare_symbols_to_ipa
from text_utils.text import change_symbols
from tqdm import tqdm


@dataclass()
class TextData:
  entry_id: int
  symbols: Symbols
  symbols_language: Language
  symbols_format: SymbolFormat


class TextDataList(GenericList[TextData]):
  def get_whole_text(self) -> str:
    texts = [''.join(x.symbols) for x in self.items()]
    res = " ".join(texts)
    return res

  def get_analytics_df(self) -> pd.DataFrame:
    values = [
      (
        entry.entry_id,
        ''.join(entry.symbols),
        len(entry.symbols),
        repr(entry.symbols_format),
        repr(entry.symbols_language),
      ) for entry in self.items()
    ]
    columns = ["Id", "Symbols", "# Symbols", "Format", "Language"]
    res = pd.DataFrame(data=values, columns=columns)
    return res

  def get_symbol_stats_df(self) -> pd.DataFrame:
    symbol_counter = Counter(symbol for item in self.items() for symbol in item.symbols)
    values = [(symbol, count) for symbol, count in symbol_counter.most_common()]
    values.sort()
    columns = ["Symbol", "# Occurrences"]
    res = pd.DataFrame(data=values, columns=columns)
    return res


def log_stats(ds_data: DsDataList, text_data: TextDataList):
  stats: List[str, int, float, float, float] = []
  text_lengths = [len(x.symbols) for x in text_data.items()]
  stats.append((
    "Overall",
    len(text_lengths),
    min(text_lengths),
    max(text_lengths),
    mean(text_lengths),
    sum(text_lengths),
  ))

  speakers_text_lengths: Dict[Speaker, List[float]] = {}
  for ds_entry, text_entry in zip(ds_data.items(), text_data.items()):
    if ds_entry.speaker_name not in speakers_text_lengths:
      speakers_text_lengths[ds_entry.speaker_name] = []
    speakers_text_lengths[ds_entry.speaker_name].append(len(text_entry.symbols))

  for speaker, speaker_text_lengths in speakers_text_lengths.items():
    stats.append((
      speaker,
      len(speaker_text_lengths),
      min(speaker_text_lengths),
      max(speaker_text_lengths),
      mean(speaker_text_lengths),
      sum(speaker_text_lengths),
    ))

  stats.sort(key=lambda x: (x[-1]), reverse=True)
  stats_csv = pd.DataFrame(stats, columns=[
    "Speaker",
    "# Entries",
    "# Min",
    "# Max",
    "# Avg",
    "# Total",
  ])

  logger = getLogger(__name__)
  with pd.option_context(
    'display.max_rows', None,
    'display.max_columns', None,
    'display.width', None,
    'display.precision', 0,
  ):
    logger.info(stats_csv)


def preprocess(data: DsDataList) -> TextDataList:
  result = TextDataList()

  for entry in data.items(True):
    text_entry = TextData(
      entry_id=entry.entry_id,
      symbols=entry.symbols,
      symbols_format=entry.symbols_format,
      symbols_language=entry.symbols_language,
    )
    result.append(text_entry)

  return result


def normalize(data: TextDataList) -> TextDataList:
  result = TextDataList()

  for entry in data.items(True):
    new_text = text_normalize(
      text=''.join(entry.symbols),
      lang=entry.symbols_language,
      text_format=entry.symbols_format,
    )

    new_symbols = text_to_symbols(
      text=new_text,
      lang=entry.symbols_language,
      text_format=entry.symbols_format,
    )

    text_entry = TextData(
      entry_id=entry.entry_id,
      symbols=new_symbols,
      symbols_format=entry.symbols_format,
      symbols_language=entry.symbols_language,
    )
    result.append(text_entry)

  return result


def convert_entry_to_ipa(entry: TextData, consider_annotations: Optional[bool], mode: Optional[EngToIPAMode], cache: LookupCache) -> TextData:
  new_symbols, new_format = symbols_to_ipa(
    symbols=entry.symbols,
    lang=entry.symbols_language,
    symbols_format=entry.symbols_format,
    mode=mode,
    consider_annotations=consider_annotations,
    cache=cache,
  )
  text_entry = TextData(
    entry_id=entry.entry_id,
    symbols=new_symbols,
    symbols_format=new_format,
    symbols_language=entry.symbols_language,
  )
  return text_entry


def convert_to_ipa(data: TextDataList, consider_annotations: Optional[bool], mode: Optional[EngToIPAMode], n_jobs: int) -> TextDataList:
  if len(data) == 0:
    return data
  # first_entry = data.items()[0]
  # prepare_symbols_to_ipa(first_entry.symbols_format, first_entry.symbols_language, mode)
  # method = partial(convert_entry_to_ipa, consider_annotations=consider_annotations,
  #                  mode=mode, cache=cache)
  # with ThreadPoolExecutor(max_workers=n_jobs) as ex:
  #   result = TextDataList(tqdm(ex.map(method, data.items()), total=len(data)))
  cache = get_empty_cache()
  result = TextDataList(
    convert_entry_to_ipa(entry, consider_annotations, mode, cache)
    for entry in data.items_tqdm()
  )
  return result


def change_ipa(data: TextDataList, ignore_tones: bool, ignore_arcs: bool, ignore_stress: bool, break_n_thongs: bool, build_n_thongs: bool) -> TextDataList:
  result = TextDataList()

  for entry in data.items(True):
    new_symbols = change_ipa_method(
      symbols=entry.symbols,
      ignore_tones=ignore_tones,
      ignore_arcs=ignore_arcs,
      ignore_stress=ignore_stress,
      break_n_thongs=break_n_thongs,
      build_n_thongs=build_n_thongs,
      language=entry.symbols_language,
    )

    text_entry = TextData(
      entry_id=entry.entry_id,
      symbols=new_symbols,
      symbols_format=entry.symbols_format,
      symbols_language=entry.symbols_language,
    )
    result.append(text_entry)

  return result


def change_text(data: TextDataList, remove_space_around_punctuation: bool) -> TextDataList:
  result = TextDataList()

  for entry in data.items():
    new_symbols = change_symbols(
      symbols=entry.symbols,
      remove_space_around_punctuation=remove_space_around_punctuation,
      lang=entry.symbols_language,
    )

    text_entry = TextData(
      entry_id=entry.entry_id,
      symbols=new_symbols,
      symbols_format=entry.symbols_format,
      symbols_language=entry.symbols_language,
    )
    result.append(text_entry)

  return result
