# -*- coding: utf-8 -*-
import re
import logging

logger = logging.getLogger(__name__)

# Minimum effective length of a sentence
MIN_SENTENCE_LENGTH = 5

# Define punctuation used for splitting sentences in Chinese
# Using Chinese full-width comma, period, question mark, exclamation mark, ellipsis
SENTENCE_TERMINATORS = "，。！？…"

# Define a regular expression for valid characters to accurately calculate length
# This includes Chinese characters, letters, and numbers.
VALID_CHAR_PATTERN = re.compile(
    r'[\u4E00-\u9FFF'  # CJK Unified Ideographs
    r'a-zA-Z'          # Half-width letters
    r'0-9'             # Half-width numbers
    r']'
)


def get_valid_text_length(sentence: str) -> int:
    """Calculates the effective length of a sentence by counting valid characters."""
    return len(VALID_CHAR_PATTERN.findall(sentence))


def split_chinese_text(long_text: str) -> list[str]:
    """
    Splits a long Chinese text into sentences based on terminators,
    and merges short sentences to prevent unnatural breaks.
    """
    if not long_text:
        return []

    # Use positive lookbehind `(?<=...)` to keep the delimiters at the end of the sentences
    raw_sentences = re.split(f'(?<=[{SENTENCE_TERMINATORS}])', long_text)
    # Strip whitespace from each sentence and filter out empty ones
    raw_sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not raw_sentences:
        return [long_text] if long_text.strip() else []

    final_sentences = []
    for sentence in raw_sentences:
        clean_len = get_valid_text_length(sentence)
        
        # If there's a previous sentence and the current one is too short, merge them.
        if final_sentences and clean_len < MIN_SENTENCE_LENGTH:
            final_sentences[-1] += sentence
        else:
            final_sentences.append(sentence)
            
    return final_sentences
