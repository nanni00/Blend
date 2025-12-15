import logging
import os
import queue
import shutil
from collections import Counter
from functools import lru_cache
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from string import ascii_lowercase
from typing import Any, Optional, Set

# The clean method is based on filtering a subset
# of tokens that may be missed as Python "None",
# and on dropping some bad characters
chars_to_replace = "\\\n\t\r'\""
translator = str.maketrans(chars_to_replace, " " * len(chars_to_replace))


@lru_cache(maxsize=1_000)
def clean(s: Any, level: int = 0, bad_tokens: Optional[Set] = {"nan", "null", "none"}):
    match level:
        case 0:
            return str(s)
        case 1:
            return str(s).lower().strip()
        case 2:
            return str(s).lower().translate(translator).strip()
        case 3:
            s = str(s).lower().translate(translator).strip()
            return s if not bad_tokens or (bad_tokens and s not in bad_tokens) else ""
        case _:
            raise ValueError(f"Clean level must be within 0 and 3: {level}")


@lru_cache(maxsize=1_000)
def calculate_xash(token: str, hash_size: int = 128) -> int:
    """
    Calculates the XASH hash of a token.
    Setting is the same as provided by XASH/MATE authors.
    """

    number_of_ones = 5
    char = list(ascii_lowercase + "0123456789" + " ")

    segment_size_dict = {64: 1, 128: 3, 256: 6, 512: 13}
    segment_size = segment_size_dict[hash_size]

    n_bits_for_chars = 37 * segment_size
    length_bit_start = n_bits_for_chars
    n_bits_for_length = hash_size - length_bit_start
    token_size = len(token)

    # - Character position encoding
    result = 0
    # Pick the 5 most infrequent characters
    counts = Counter(token).items()
    sorted_counts = sorted(counts, key=lambda char_occurances: char_occurances[::-1])
    selected_chars = [char for char, _ in sorted_counts[:number_of_ones]]
    # Encode the position of the 5 most infrequent characters
    for c in selected_chars:
        if c not in char:
            continue
        # Calculate the mean position of the character and set the one bit in the corresponding segment
        indices = [i for i, ltr in enumerate(token) if ltr == c]
        mean_index = sum(indices) / len(indices)
        normalized_mean_index = mean_index / token_size
        segment = max(int(normalized_mean_index * segment_size - 1e-6), 0)  # Legacy fix
        location = char.index(c) * segment_size + segment
        result = result | 2**location

    # Rotate position encoding
    shift_distance = (
        length_bit_start
        * (token_size % (hash_size - length_bit_start))
        // (hash_size - length_bit_start)
    )
    left_bits = result << shift_distance
    wrapped_bits = result >> (n_bits_for_chars - shift_distance)
    cut_overlapping_bits = 2**n_bits_for_chars

    result = (left_bits | wrapped_bits) % cut_overlapping_bits

    # - Add length bit
    length_bit = 1 << (length_bit_start + token_size % n_bits_for_length)
    result = result | length_bit

    return result


def init_logger(log_directory: Path) -> tuple[logging.Logger, QueueListener]:
    root = logging.getLogger(f"blend_logger_{os.getpid()}")
    root.setLevel(logging.INFO)
    q = queue.Queue(-1)
    queue_handler = QueueHandler(q)
    if root.hasHandlers():
        root.handlers.clear()

    old_dirs = sorted([d for d in os.listdir(log_directory.parent)], reverse=True)
    dirs_to_delete = old_dirs[3:] if len(old_dirs) > 3 else []

    for dir_to_delete in dirs_to_delete:
        dir_path = log_directory.parent.joinpath(dir_to_delete)
        shutil.rmtree(dir_path)

    if not root.hasHandlers():
        logfile = log_directory.joinpath(f"{os.getpid()}.log")
        handler = RotatingFileHandler(logfile, mode="a", maxBytes=1024**3)
        log_formatter = logging.Formatter(
            "[%(asctime)s][%(process)d][%(threadName)s][%(levelname)s],%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(log_formatter)
        root.addHandler(queue_handler)

    listener = QueueListener(q, handler)
    return root, listener
