# -*- coding: utf-8 -*-
import re
import pypinyin
from pypinyin.style._utils import get_initials, get_finals
from typing import List
from .SymbolsV2 import symbols_v2, symbol_to_id_v2

# Regular expression to match Chinese characters
_CHINESE_CHARACTERS_RE = re.compile(r'[\u4e00-\u9fa5]')

# Regular expression to match punctuation and symbols that act as delimiters.
# Using a capturing group `()` to keep the delimiters in the split result.
_PUNCTUATION_RE = re.compile(r'([,./?!~…・;:"\'\n\t 、，。！？；：]+)')

class ChineseG2P:
    """
    A Grapheme-to-Phoneme (G2P) converter for Chinese text, designed to be
    consistent with the Japanese G2P implementation.
    """

    @staticmethod
    def _post_replace_phoneme(ph: str) -> str:
        """Post-processes a single phoneme or punctuation mark for normalization."""
        rep_map = {
            "：": ",", "；": ",", "，": ",", "。": ".",
            "！": "!", "？": "?", "\n": ".", "·": ",",
            "、": ",", "...": "…",
        }
        return rep_map.get(ph, ph)

    @staticmethod
    def _is_chinese(text: str) -> bool:
        """Check if the text contains Chinese characters."""
        return _CHINESE_CHARACTERS_RE.search(text) is not None

    @staticmethod
    def _get_phonemes(text: str) -> List[str]:
        """Convert a single Chinese text segment to phonemes using pypinyin."""
        phonemes = []
        # Use TONE3 style for tone numbers, and neutral_tone_with_five for light tones
        pinyins = pypinyin.pinyin(text, style=pypinyin.Style.TONE3, neutral_tone_with_five=True)
        
        for p_list in pinyins:
            syllable_with_tone = p_list[0]
            
            # Separate syllable and tone
            syllable = syllable_with_tone.rstrip('12345')
            tone = syllable_with_tone[len(syllable):]

            # Ensure there's a tone (default to 5 for neutral)
            if not tone:
                tone = '5'

            # Decompose into initials and finals
            initial = get_initials(syllable, strict=False)
            final = get_finals(syllable, strict=False)

            if initial:
                phonemes.append(initial)
            if final:
                # Attach tone to the final
                phonemes.append(f"{final}{tone}")
        
        return phonemes

    @staticmethod
    def g2p(text: str) -> List[str]:
        """
        Converts Chinese text to a sequence of phonemes, preserving punctuation.
        """
        if not text.strip():
            return []

        result_phonemes = []
        # Split text by punctuation while keeping the delimiters
        segments = _PUNCTUATION_RE.split(text)

        for segment in segments:
            if not segment:
                continue
            
            # If the segment is purely punctuation, add it to the list
            if _PUNCTUATION_RE.fullmatch(segment):
                result_phonemes.append(segment.strip())
            # Otherwise, process the segment for phonemes
            else:
                phonemes = ChineseG2P._get_phonemes(segment)
                result_phonemes.extend(phonemes)
        
        # Apply post-replacement to each phoneme/punctuation mark
        processed_phonemes = [ChineseG2P._post_replace_phoneme(p) for p in result_phonemes]
        
        # Filter out any empty strings that might have been added
        return [p for p in processed_phonemes if p]


def chinese_to_phones(text: str) -> list[int]:
    """
    Converts a Chinese text string to a list of phoneme IDs.
    """
    phones = ChineseG2P.g2p(text)
    
    valid_phones = []
    for ph in phones:
        if ph in symbol_to_id_v2:
            valid_phones.append(ph)
        else:
            # Fallback for unknown phonemes/symbols
            valid_phones.append("UNK")
            
    ids = [symbol_to_id_v2[ph] for ph in valid_phones]
    return ids
