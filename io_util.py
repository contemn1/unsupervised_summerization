import hashlib
import io
import json
import logging
import os
import re
import sys
from shutil import copyfile

import numpy as np
from rouge import Rouge


def output_iterator(file_path, output_list, process=lambda x: x):
    try:
        with io.open(file_path, mode="w+", encoding="utf-8") as file:
            for line in output_list:
                file.write(process(line) + "\n")
    except IOError as error:
        logging.error("Failed to open file {0}".format(error))
        sys.exit(1)


def read_file(file_path, encoding="utf-8", preprocess=lambda x: x.strip()):
    try:
        with io.open(file_path, encoding=encoding) as file:
            for sentence in file.readlines():
                yield (preprocess(sentence))

    except IOError as err:
        logging.error("Failed to open file {0}".format(err))
        sys.exit(1)


def has_hex(s: str):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    if isinstance(s, str) and (sys.version_info > (3, 0)):
        s = s.encode("utf-8")
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()


def generate_test_set(file_path, cnn_dir, daily_mail_dir, output_dir):
    file_list = read_file(file_path)
    url_hashes = get_url_hashes(file_list)
    story_fnames = [s + ".story" for s in url_hashes]
    for name in story_fnames:
        cnn_path = os.path.join(cnn_dir, name)
        daily_mail_path = os.path.join(daily_mail_dir, name)
        output_path = os.path.join(output_dir, name)
        if os.path.isfile(cnn_path):
            copyfile(cnn_path, output_path)
        elif os.path.isfile(daily_mail_path):
            copyfile(daily_mail_path, output_path)


def split_file(file_path, output_dir, preprocess=lambda x: x.strip(), chunk_size=2000):
    input_list = list(read_file(file_path, preprocess=preprocess))
    input_base_name = os.path.basename(file_path)
    txt_suffix = re.compile(r"\.txt")
    if txt_suffix.search(input_base_name):
        input_base_name = txt_suffix.sub("", input_base_name)
    output_template = "{0}_part{1}.txt"
    for idx in range(0, len(input_list), chunk_size):
        partial_list = input_list[idx: idx + chunk_size]
        output_path = os.path.join(output_dir, output_template.format(input_base_name, idx // chunk_size))
        output_iterator(output_path, partial_list)


def select_most_similar_sentences(tup):
    doc_list, summary_list = tup
    doc_list = [ele for ele in doc_list if ele != "."]
    doc_number = len(doc_list)
    rouge_calculator = Rouge(metrics=["rouge-1"])
    summary_list = [[ele] * doc_number for ele in summary_list]
    max_idx_list = []
    for idx, sum in enumerate(summary_list):
        try:
            scores = [ele["rouge-1"]["f"] for ele in rouge_calculator.get_scores(doc_list, sum)]
            max_idx = np.argmax(scores)
            max_idx_list.append(max_idx)

        except ValueError as err:
            print(doc_list)
            print(sum)

    return max_idx_list


if __name__ == '__main__':
    input_dir = "/home/zxj/Downloads/cnn_dm_test/2part"
    part_summary = os.path.join(input_dir,
                                "smry_cnn_dm_input_truncated_all_Ks10_clust1_eosavg0_n6_ns10_nf300_K4_a0.1_b0.7_all.txt")
    part_map_file = os.path.join(input_dir, "partition_map")
    summary_list = read_file(part_summary)
    with open(part_map_file, mode="r") as part_map:
        part_dict = json.load(part_map)
        part_dict = {int(key): value for key, value in part_dict.items()}

    length = max(part_dict.values()) + 1
    new_result_list = [[] for _ in range(length)]
    for idx, line in enumerate(summary_list):
        new_result_list[part_dict[idx]].append(line)
    
    output_iterator(os.path.join(input_dir, "smry_cnn_dm_generated.txt"),new_result_list,  process=lambda x: " ".join(x))

