import os
from collections import deque
from typing import Iterable
from typing import List
from io_util import read_file
from transformers import GPT2Tokenizer


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
            result_list = [(tokenize_list(doc, self.tokenizer), tokenize_list(summary, self.tokenizer)) for doc, summary in result_list]
        return result_list

