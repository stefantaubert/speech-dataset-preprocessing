from functools import partial
from logging import getLogger
from pathlib import Path
from shutil import rmtree
from typing import Callable, Optional

from speech_dataset_preprocessing.app.ds import get_ds_dir, load_ds_data
from speech_dataset_preprocessing.core.text import (SymbolsDict, TextDataList,
                                                    change_ipa, convert_to_ipa,
                                                    log_stats, normalize,
                                                    preprocess)
from speech_dataset_preprocessing.globals import DEFAULT_CSV_SEPERATOR
from speech_dataset_preprocessing.utils import (get_subdir, load_obj, save_obj,
                                                save_txt)
from text_utils import EngToIPAMode

_text_data_csv = "data.pkl"
_text_symbols_json = "symbols.json"
_whole_text_txt = "text.txt"
ANALYSIS_DF_PATH = "analysis.csv"


def _get_text_root_dir(ds_dir: Path, create: bool = False) -> Path:
  return get_subdir(ds_dir, "text", create)


def get_text_dir(ds_dir: Path, text_name: str, create: bool = False) -> Path:
  return get_subdir(_get_text_root_dir(ds_dir, create), text_name, create)


def load_text_symbols_json(text_dir: Path) -> SymbolsDict:
  path = text_dir / _text_symbols_json
  return SymbolsDict.load(path)


def save_text_symbols_json(text_dir: Path, data: SymbolsDict) -> None:
  path = text_dir / _text_symbols_json
  data.save(path)


def save_whole_text(text_dir: Path, data: TextDataList) -> None:
  path = text_dir / _whole_text_txt
  text = data.get_whole_text()
  save_txt(path, text)


def save_analytics_df(text_dir: Path, data: TextDataList) -> None:
  path = text_dir / ANALYSIS_DF_PATH
  analytics_df = data.get_analytics_df()
  analytics_df.to_csv(path, sep=DEFAULT_CSV_SEPERATOR, header=True, index=False)


def load_text_data(text_dir: Path) -> TextDataList:
  path = text_dir / _text_data_csv
  return load_obj(path)


def save_text_data(text_dir: Path, data: TextDataList) -> None:
  path = text_dir / _text_data_csv
  save_obj(data, path)


def text_stats(base_dir: Path, ds_name: str, text_name: str):
  logger = getLogger(__name__)
  logger.info(f"Stats of {text_name}")
  ds_dir = get_ds_dir(base_dir, ds_name)
  text_dir = get_text_dir(ds_dir, text_name)
  if text_dir.is_dir():
    ds_data = load_ds_data(ds_dir)
    text_data = load_text_data(text_dir)
    log_stats(ds_data, text_data)


def export_text(base_dir: Path, ds_name: str, text_name: str) -> None:
  logger = getLogger(__name__)
  logger.info("Exporting text...")
  ds_dir = get_ds_dir(base_dir, ds_name)
  text_dir = get_text_dir(ds_dir, text_name)
  if text_dir.is_dir():
    text_data = load_text_data(text_dir)
    save_whole_text(text_dir, text_data)
    save_analytics_df(text_dir, text_data)
    logger.info("Finished.")


def preprocess_text(base_dir: Path, ds_name: str, text_name: str, overwrite: bool) -> None:
  logger = getLogger(__name__)
  logger.info("Preprocessing text...")
  ds_dir = get_ds_dir(base_dir, ds_name)
  text_dir = get_text_dir(ds_dir, text_name)
  if text_dir.is_dir() and not overwrite:
    logger.error("Already exists.")
    return

  data = load_ds_data(ds_dir)
  text_data = preprocess(data)

  if text_dir.is_dir():
    assert overwrite
    logger.info("Overwriting existing data.")
    rmtree(text_dir)
  text_dir.mkdir(parents=True, exist_ok=False)

  save_text_data(text_dir, text_data)
  save_text_symbols_json(text_dir, text_data.get_symbol_stats())
  save_whole_text(text_dir, text_data)
  save_analytics_df(text_dir, text_data)


def _text_op(base_dir: Path, ds_name: str, orig_text_name: str, dest_text_name: str, operation: Callable[[TextDataList], TextDataList], overwrite: bool):
  logger = getLogger(__name__)
  ds_dir = get_ds_dir(base_dir, ds_name)
  orig_text_dir = get_text_dir(ds_dir, orig_text_name)
  assert orig_text_dir.is_dir()
  dest_text_dir = get_text_dir(ds_dir, dest_text_name)
  if dest_text_dir.is_dir() and not overwrite:
    logger.error("Already exists.")
    return

  logger.info("Reading data...")
  data = load_text_data(orig_text_dir)
  text_data = operation(data)

  if dest_text_dir.is_dir():
    assert overwrite
    logger.info("Overwriting existing data.")
    rmtree(dest_text_dir)
  dest_text_dir.mkdir(parents=True, exist_ok=False)

  save_text_data(dest_text_dir, text_data)
  save_text_symbols_json(dest_text_dir, text_data.get_symbol_stats())
  save_whole_text(dest_text_dir, text_data)
  save_analytics_df(dest_text_dir, text_data)
  logger.info("Dataset processed.")


def text_normalize(base_dir: Path, ds_name: str, orig_text_name: str, dest_text_name: str, overwrite: bool) -> None:
  logger = getLogger(__name__)
  logger.info("Normalizing text...")
  operation = partial(normalize)
  _text_op(base_dir, ds_name, orig_text_name, dest_text_name, operation, overwrite)


def text_convert_to_ipa(base_dir: Path, ds_name: str, orig_text_name: str, dest_text_name: str, consider_ipa_annotations: Optional[bool], mode: Optional[EngToIPAMode], overwrite: bool) -> None:
  logger = getLogger(__name__)
  logger.info("Converting text to IPA...")
  operation = partial(
    convert_to_ipa,
    mode=mode,
    consider_ipa_annotations=consider_ipa_annotations,
  )
  _text_op(base_dir, ds_name, orig_text_name, dest_text_name, operation, overwrite)


def text_change_ipa(base_dir: Path, ds_name: str, orig_text_name: str, dest_text_name: str, ignore_tones: bool, ignore_arcs: bool, ignore_stress: bool, overwrite: bool) -> None:
  logger = getLogger(__name__)
  logger.info("Changing IPA...")
  operation = partial(
    change_ipa,
    ignore_tones=ignore_tones,
    ignore_arcs=ignore_arcs,
    ignore_stress=ignore_stress,
  )
  _text_op(base_dir, ds_name, orig_text_name, dest_text_name, operation, overwrite)
