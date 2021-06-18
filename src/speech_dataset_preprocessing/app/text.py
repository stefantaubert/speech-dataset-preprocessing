import os
from functools import partial
from logging import getLogger
from typing import Optional

from speech_dataset_preprocessing.app.ds import (get_ds_dir, load_ds_csv,
                                                 load_ds_symbols_json)
from speech_dataset_preprocessing.core.text import (SymbolsDict, TextData,
                                                    TextDataList,
                                                    convert_to_ipa, log_stats,
                                                    normalize, preprocess)
from speech_dataset_preprocessing.utils import get_subdir, save_txt
from text_utils import EngToIpaMode, SymbolIdDict

_text_data_csv = "data.csv"
_text_symbols_json = "symbols.json"
_whole_text_txt = "text.txt"
_text_symbol_ids_json = "symbol_ids.json"


def _get_text_root_dir(ds_dir: str, create: bool = False):
  return get_subdir(ds_dir, "text", create)


def get_text_dir(ds_dir: str, text_name: str, create: bool = False):
  return get_subdir(_get_text_root_dir(ds_dir, create), text_name, create)


def load_text_symbol_converter(text_dir: str) -> SymbolIdDict:
  path = os.path.join(text_dir, _text_symbol_ids_json)
  return SymbolIdDict.load_from_file(path)


def save_text_symbol_converter(text_dir: str, data: SymbolIdDict):
  path = os.path.join(text_dir, _text_symbol_ids_json)
  data.save(path)


def load_text_symbols_json(text_dir: str) -> SymbolsDict:
  path = os.path.join(text_dir, _text_symbols_json)
  return SymbolsDict.load(path)


def save_text_symbols_json(text_dir: str, data: SymbolsDict):
  path = os.path.join(text_dir, _text_symbols_json)
  data.save(path)


def save_whole_text(text_dir: str, data: TextDataList):
  path = os.path.join(text_dir, _whole_text_txt)
  text = data.get_whole_text()
  save_txt(path, text)


def load_text_csv(text_dir: str) -> TextDataList:
  path = os.path.join(text_dir, _text_data_csv)
  return TextDataList.load(TextData, path)


def save_text_csv(text_dir: str, data: TextDataList):
  path = os.path.join(text_dir, _text_data_csv)
  data.save(path)


def text_stats(base_dir: str, ds_name: str, text_name: str):
  logger = getLogger(__name__)
  logger.info(f"Stats of {text_name}")
  ds_dir = get_ds_dir(base_dir, ds_name)
  text_dir = get_text_dir(ds_dir, text_name)
  if os.path.isdir(text_dir):
    ds_data = load_ds_csv(ds_dir)
    text_data = load_text_csv(text_dir)
    log_stats(ds_data, text_data, logger)


def export_text(base_dir: str, ds_name: str, text_name: str):
  logger = getLogger(__name__)
  logger.info("Exporting text...")
  ds_dir = get_ds_dir(base_dir, ds_name)
  text_dir = get_text_dir(ds_dir, text_name)
  if os.path.isdir(text_dir):
    data = load_text_csv(text_dir)
    save_whole_text(text_dir, data)
    logger.info("Finished.")


def preprocess_text(base_dir: str, ds_name: str, text_name: str):
  logger = getLogger(__name__)
  logger.info("Preprocessing text...")
  ds_dir = get_ds_dir(base_dir, ds_name)
  text_dir = get_text_dir(ds_dir, text_name)
  if os.path.isdir(text_dir):
    logger.error("Already exists.")
  else:
    data = load_ds_csv(ds_dir)
    symbol_ids = load_ds_symbols_json(ds_dir)
    text_data, conv, all_symbols = preprocess(data, symbol_ids)
    os.makedirs(text_dir)
    save_text_csv(text_dir, text_data)
    save_text_symbol_converter(text_dir, conv)
    save_text_symbols_json(text_dir, all_symbols)


def _text_op(base_dir: str, ds_name: str, orig_text_name: str, dest_text_name: str, operation):
  logger = getLogger(__name__)
  ds_dir = get_ds_dir(base_dir, ds_name)
  orig_text_dir = get_text_dir(ds_dir, orig_text_name)
  assert os.path.isdir(orig_text_dir)
  dest_text_dir = get_text_dir(ds_dir, dest_text_name)
  if os.path.isdir(dest_text_dir):
    logger.error("Already exists.")
  else:
    logger.info("Reading data...")
    data = load_text_csv(orig_text_dir)
    orig_conv = load_text_symbol_converter(orig_text_dir)
    text_data, conv, all_symbols = operation(data, orig_conv)
    os.makedirs(dest_text_dir)
    save_text_csv(dest_text_dir, text_data)
    save_text_symbol_converter(dest_text_dir, conv)
    save_text_symbols_json(dest_text_dir, all_symbols)
    logger.info("Dataset processed.")


def text_normalize(base_dir: str, ds_name: str, orig_text_name: str, dest_text_name: str):
  logger = getLogger(__name__)
  logger.info("Normalizing text...")
  operation = partial(
    normalize,
    logger=logger,
  )
  _text_op(base_dir, ds_name, orig_text_name, dest_text_name, operation)


def text_convert_to_ipa(base_dir: str, ds_name: str, orig_text_name: str, dest_text_name: str, ignore_tones: bool = False, ignore_arcs: bool = False, consider_ipa_annotations: bool = False, mode: Optional[EngToIpaMode] = None):
  logger = getLogger(__name__)
  logger.info("Converting text to IPA...")
  operation = partial(
    convert_to_ipa,
    ignore_tones=ignore_tones,
    ignore_arcs=ignore_arcs,
    mode=mode,
    consider_ipa_annotations=consider_ipa_annotations,
    logger=logger
  )
  _text_op(base_dir, ds_name, orig_text_name, dest_text_name, operation)
