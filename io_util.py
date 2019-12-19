import hashlib
import io
import logging
import os
import sys
from shutil import copyfile


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
    story_fnames = [s+".story" for s in url_hashes]
    for name in story_fnames:
        cnn_path = os.path.join(cnn_dir, name)
        daily_mail_path = os.path.join(daily_mail_dir, name)
        output_path = os.path.join(output_dir, name)
        if os.path.isfile(cnn_path):
            copyfile(cnn_path, output_path)
        elif os.path.isfile(daily_mail_path):
            copyfile(daily_mail_path, output_path)
