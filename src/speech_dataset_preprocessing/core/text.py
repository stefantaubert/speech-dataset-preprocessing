from dataclasses import dataclass
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import pandas as pd
from numpy.core.fromnumeric import mean
from speech_dataset_preprocessing.core.ds import DsDataList
from speech_dataset_preprocessing.utils import GenericList, get_counter
from text_utils import (EngToIPAMode, Language, SymbolIdDict, Symbols,
                        SymbolsDict, chn_to_ipa, deserialize_list, eng_to_ipa,
                        ger_to_ipa, remove_arcs, remove_stress, remove_tones,
                        text_normalize, text_to_symbols)


@dataclass()
class TextData:
  entry_id: int
  text: str
  serialized_symbol_ids: str
  lang: Language

  def load_init(self):
    self.lang = Language(self.lang)


class TextDataList(GenericList[TextData]):
  def load_init(self):
    for item in self.items():
      item.load_init()

  def get_whole_text(self) -> str:
    texts = [x.text for x in self.items()]
    res = " ".join(texts)
    return res


def log_stats(ds_data: DsDataList, text_data: TextDataList):
  stats: List[str, int, float, float, float] = []
  text_lengths = [len(deserialize_list(x.serialized_symbol_ids)) for x in text_data.items()]
  stats.append((
    "Overall",
    len(text_lengths),
    min(text_lengths),
    max(text_lengths),
    mean(text_lengths),
    sum(text_lengths),
  ))

  speakers_text_lengths: List[int, List[float]] = {}
  speaker_names: Dict[int, str] = {}
  for ds_entry, text_entry in zip(ds_data.items(), text_data.items()):
    if ds_entry.speaker_id not in speakers_text_lengths:
      speakers_text_lengths[ds_entry.speaker_id] = []
      speaker_names[ds_entry.speaker_id] = ds_entry.speaker_name
    speakers_text_lengths[ds_entry.speaker_id].append(
      len(deserialize_list(text_entry.serialized_symbol_ids)))

  for k, speaker_text_lengths in speakers_text_lengths.items():
    stats.append((
      f"{speaker_names[k]} ({k})",
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

  with pd.option_context(
    'display.max_rows', None,
    'display.max_columns', None,
    'display.width', None,
    'display.precision', 0,
  ):
    print(stats_csv)


def convert_to_ipa(data: TextDataList, symbol_converter: SymbolIdDict, ignore_tones: bool, ignore_arcs: bool, ignore_stress: bool, mode: Optional[EngToIPAMode], consider_ipa_annotations: bool) -> Tuple[TextDataList, SymbolIdDict, SymbolsDict]:
  processed_data: List[Tuple[int, Symbols, Language]] = []

  for values in data.items(True):
    current_text = symbol_converter.get_text(values.serialized_symbol_ids)
    if values.lang == Language.ENG:
      if mode is None:
        ex = "Please specify the IPA conversion mode."
        logger = getLogger(__name__)
        logger.exception(ex)
        raise Exception(ex)
      new_symbols = eng_to_ipa(current_text, consider_ipa_annotations, mode=mode)
    elif values.lang == Language.GER:
      new_symbols = ger_to_ipa(current_text, consider_ipa_annotations)
    elif values.lang == Language.CHN:
      new_symbols = chn_to_ipa(current_text, consider_ipa_annotations)
    elif values.lang == Language.IPA:
      new_symbols = symbol_converter.get_symbols(values.serialized_symbol_ids)
    else:
      assert False

    if ignore_arcs:
      new_symbols = remove_arcs(new_symbols)

    if ignore_tones:
      new_symbols = remove_tones(new_symbols)

    if ignore_stress:
      new_symbols = remove_stress(new_symbols)

    processed_data.append((values.entry_id, new_symbols, Language.IPA))

  return _prepare_data(processed_data)


def normalize(data: TextDataList, symbol_converter: SymbolIdDict) -> Tuple[TextDataList, SymbolIdDict, SymbolsDict]:
  processed_data: List[Tuple[int, Symbols, Language]] = []

  for values in data.items(True):
    new_text = text_normalize(
      text=symbol_converter.get_text(values.serialized_symbol_ids),
      lang=values.lang,
    )
    new_symbols = text_to_symbols(
      text=new_text,
      lang=values.lang,
    )

    processed_data.append((values.entry_id, new_symbols, values.lang))

  return _prepare_data(processed_data)


def preprocess(data: DsDataList, symbol_ids: SymbolIdDict) -> Tuple[TextDataList, SymbolIdDict, SymbolsDict]:
  processed_data: List[Tuple[int, Symbols, Language]] = []

  for values in data.items(True):
    symbols = symbol_ids.get_symbols(values.serialized_symbols)
    processed_data.append((values.entry_id, symbols, values.lang))

  return _prepare_data(processed_data)


def _prepare_data(processed_data: List[Tuple[int, List[str], List[int], Language]]) -> Tuple[TextDataList, SymbolIdDict, SymbolsDict]:
  result = TextDataList()
  symbol_counter = get_counter([x[1] for x in processed_data])
  symbols_dict = SymbolsDict.fromcounter(symbol_counter)
  conv = SymbolIdDict.init_from_symbols(set(symbols_dict.keys()))

  for entry_id, symbols, lang in processed_data:
    text = SymbolIdDict.symbols_to_text(symbols)
    serialized_symbol_ids = conv.get_serialized_ids(symbols)
    data = TextData(entry_id, text, serialized_symbol_ids, lang)
    result.append(data)

  return result, conv, symbols_dict
