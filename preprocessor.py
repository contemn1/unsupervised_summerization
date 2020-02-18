import json
import os
import re
from collections import deque
from typing import Iterable
from typing import List

from spacy.lang.en import English
from transformers import GPT2Tokenizer

from io_util import read_file, output_iterator, split_file
from bisect import bisect_left

def tokenize_list(content_list: List[str], tokenizer: GPT2Tokenizer):
    return [tokenizer.tokenize(sentence) for sentence in content_list]


class Preprocessor(object):
    def __init__(self, target_dir, tokenizer):
        self.target_dir = target_dir
        self.tokenizer = tokenizer
        self.tokenized = False

    def __fix_missing_period(self, line: str) -> str:

        dm_single_close_quote = u'\u2019'  # unicode
        dm_double_close_quote = u'\u201d'
        END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
                      ")"]  # acceptable ways to end a sentence

        """Adds a period to a line that is missing a period"""
        if "@highlight" in line: return line
        if line == "":
            return line
        if line[-1] in END_TOKENS:
            return line
        # print line[-1]
        return line + " ."

    def __process_story(self, raw_story: Iterable[str]):
        """ Extract the story and summary from a story file.
        Attributes:
            raw_story (str): content of the story file as an utf-8 encoded string.
        Raises:
            IndexError: If the stoy is empty or contains no highlights.
        """
        nonempty_lines = filter(lambda x: len(x) != 0, raw_story)

        # for some unknown reason some lines miss a period, add it
        nonempty_lines = [self.__fix_missing_period(line) for line in nonempty_lines]

        # gather article lines
        story_lines = []
        lines = deque(nonempty_lines)
        while True:
            try:
                element = lines.popleft()
                if element.startswith("@highlight"):
                    break
                story_lines.append(element)
            except IndexError:
                # if "@highlight" is absent from the file we pop
                # all elements until there is None.
                return story_lines, []

        # gather summary lines
        summary_lines = list(filter(lambda t: not t.startswith("@highlight"), lines))

        return story_lines, summary_lines

    def get_document_summary(self, tokenize=False):
        root_dir = self.target_dir
        file_path_list = [file_name for file_name in os.listdir(root_dir) if file_name[-6:] == ".story"]
        result_list = (read_file(os.path.join(root_dir, path)) for path in file_path_list)
        result_list = [self.__process_story(doc) for doc in result_list]
        result_list = [tup for tup in result_list if tup[0] and tup[1]]

        if tokenize:
            self.tokenized = tokenize
            result_list = [(tokenize_list(doc, self.tokenizer), tokenize_list(summary, self.tokenizer)) for doc, summary
                           in result_list]
        return result_list


def chunks(input_list, number):
    size = len(input_list) / number
    if size > len(input_list) // number:
        size = int(size) + 1
    else:
        size = int(size)
    for idx in range(0, len(input_list), size):
        yield input_list[idx: idx + size]


def preprocess_file(test_dir, summary_path):
    cnn_preprocessor = Preprocessor(test_dir, tokenizer=None)
    tokenize = False
    content_list = cnn_preprocessor.get_document_summary(tokenize)
    doc_list, summary_list = zip(*content_list)
    nlp = English()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)
    cnn_pattern = re.compile(r"\(CNN\)")
    #words = ([ele.text for sent in doc for ele in nlp(cnn_pattern.sub("", sent)).sents] for doc in doc_list)
    summaries = [[" ".join([ele.text for ele in tokenizer(sent)]) for sent in sum] for sum in summary_list]
    #output_iterator(os.path.join(output_path, "cnn_dm_input.txt"), words, process=lambda x: "\001".join(x))
    output_iterator(summary_path, summaries, process=lambda x: "\001".join(x))


def partition_documents(doc_list, num_partitions):
    counter = 0
    new_sentence_list = []
    partition_map = dict()
    for idx, ele in enumerate(doc_list):
        if len(ele) < num_partitions:
            new_sentence_list.append(" ".join(ele))
            partition_map[counter] = idx
            counter += 1
            continue

        for part in chunks(ele, num_partitions):
            new_sentence_list.append(" ".join(part))
            partition_map[counter] = idx
            counter += 1
    return new_sentence_list, partition_map


def merge_partition(partition_map, input_iter):
    max_length = max(partition_map.values()) + 1
    new_result_list = ["" for _ in range(max_length)]
    for idx, ele in enumerate(input_iter):
        new_result_list[partition_map[str(idx)]] += ele
        new_result_list[partition_map[str(idx)]] += " "
    return new_result_list

if __name__ == '__main__':
    input_dir = "/home/zxj/Documents/github/PacSum/extracted_parts/extracted_contents_all.txt"
    output_template = "/home/zxj/Documents/github/PacSum/extracted_parts/content_part_{0}.txt"
    doc_list = list(read_file(input_dir, preprocess=lambda x: x.strip()))
    idx = 0
    for ele in chunks(doc_list, 7):
        output_iterator(output_template.format(idx), ele)
        idx += 1