import os
import json
import errno
from shutil import rmtree
from os.path import join as pjoin

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
        json.dump(json_obj, make_file, indent="\t")
    return

def write_jsonl(save_path, json_obj):
    with open(save_path, 'w', encoding='utf-8') as f:
        for entry in json_obj:
            json.dump(entry, f)
            f.write('\n')
    return

def split_list_into_chunks(lst, chunk_size=2):
    """
	Split a list into chunks of the specified size.

    Parameters:
    - lst: The list to be split.
    - chunk_size: The size of each chunk.

    Returns:
    A list of chunks, where each chunk is a sublist of the original list.
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def load_prompt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
