import os
import re
import json
import errno
from shutil import rmtree

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
def del_folder(path):
    try:
        rmtree(path)
    except:
        pass

def read_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def write_json(save_path, json_obj):
    with open(save_path, 'w', encoding='utf-8') as make_file:
        json.dump(json_obj, make_file, indent="\t", ensure_ascii=False)
    return

def write_jsonl(save_path, json_obj):
    with open(save_path, 'w', encoding='utf-8') as f:
        for entry in json_obj:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    return

def write_line(path, entry):
    with open(path, 'a+') as f:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')