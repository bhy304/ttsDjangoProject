from multiprocessing import Pool
from contextlib import closing
from tqdm import tqdm
import json
import os
import re
import sys
from datetime import datetime, timedelta

class ValueWindow():
  def __init__(self, window_size=100):
    self._window_size = window_size
    self._values = []

  def append(self, x):
    self._values = self._values[-(self._window_size - 1):] + [x]

  @property
  def sum(self):
    return sum(self._values)

  @property
  def count(self):
    return len(self._values)

  @property
  def average(self):
    return self.sum / max(1, self.count)

  def reset(self):
    self._values = []

def parallel_run(fn, items, desc="", parallel=True):
    results = []

    if parallel:
        with closing(Pool()) as pool:
            for out in tqdm(pool.imap_unordered(
                    fn, items), total=len(items), desc=desc):
                if out is not None:
                    results.append(out)
    else:
        for item in tqdm(items, total=len(items), desc=desc):
            out = fn(item)
            if out is not None:
                results.append(out)

    return results

def add_postfix(path, postfix):
    path_without_ext, ext = path.rsplit('.', 1)
    return "{}.{}.{}".format(path_without_ext, postfix, ext)

def write_json(path, data):
    with open(path, 'w',encoding='utf-8') as f:
        json.dump(data, f, indent=4, sort_keys=True, ensure_ascii=False)

def remove_file(path):
    if os.path.exists(path):
        print(" [*] Removed: {}".format(path))
        os.remove(path)

def backup_file(path):
    root, ext = os.path.splitext(path)
    new_path = "{}.backup_{}{}".format(root, get_time(), ext)

    os.rename(path, new_path)
    print(" [*] {} has backup: {}".format(path, new_path))

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

