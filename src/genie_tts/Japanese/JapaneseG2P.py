# -*- coding: utf-8 -*-
"""
A universal G2P module that automatically handles Japanese and Chinese.
"""
import re
import pyopenjtalk
import pypinyin
from pypinyin.style._utils import get_initials, get_finals
from typing import List
from .SymbolsV2 import symbols_v2, symbol_to_id_v2

# --- Language Detection ---
_CHINESE_CHARACTERS_RE = re.compile(r'[\u4e00-\u9fa5]')

def is_chinese(text: str) -> bool:
    """Check if the text contains Chinese characters."""
    return _CHINESE_CHARACTERS_RE.search(text) is not None

# --- Japanese G2P Constants ---
_CONSECUTIVE_PUNCTUATION_RE = re.compile(r"([,./?!~…・])\1+")
_SYMBOLS_TO_JAPANESE = [
    (re.compile("%"), "パーセント"),
    (re.compile("％"), "パーセント"),
]
_JAPANESE_CHARACTERS_RE = re.compile(
    r"[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)
_JAPANESE_MARKS_RE = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

class JapaneseG2P:
    """
    A universal G2P converter that automatically handles Japanese and Chinese.
    """
    @staticmethod
    def _text_normalize(text: str) -> str:
        for regex, replacement in _SYMBOLS_TO_JAPANESE:
            text = re.sub(regex, replacement, text)
        text = _CONSECUTIVE_PUNCTUATION_RE.sub(r"\1", text)
        text = text.lower()
        return text

    @staticmethod
    def _post_replace_phoneme(ph: str) -> str:
        rep_map = {
            "：": ",", "；": ",", "，": ",", "。": ".",
            "！": "!", "？": "?", "\n": ".", "·": ",",
            "、": ",", "...": "…",
        }
        return rep_map.get(ph, ph)

    @staticmethod
    def _numeric_feature_by_regex(regex: str, s: str) -> int:
        match = re.search(regex, s)
        return int(match.group(1)) if match else -50

    @staticmethod
    def _pyopenjtalk_g2p_prosody(text: str) -> List[str]:
        labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
        phones = []
        for n, lab_curr in enumerate(labels):
            p3 = re.search(r"-(.*?)\+", lab_curr).group(1)
            if p3 in "AEIOU":
                p3 = p3.lower()

            if p3 == "sil":
                if n == 0:
                    phones.append("^")
                elif n == len(labels) - 1:
                    e3 = JapaneseG2P._numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                    phones.append("?" if e3 == 1 else "$")
                continue
            elif p3 == "pau":
                phones.append("_")
                continue
            else:
                phones.append(p3)

            a1 = JapaneseG2P._numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
            a2 = JapaneseG2P._numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
            a3 = JapaneseG2P._numeric_feature_by_regex(r"\+(\d+)/", lab_curr)
            f1 = JapaneseG2P._numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)
            lab_next = labels[n + 1] if n + 1 < len(labels) else ""
            a2_next = JapaneseG2P._numeric_feature_by_regex(r"\+(\d+)\+", lab_next)

            if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
                phones.append("#")
            elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
                phones.append("]")
            elif a2 == 1 and a2_next == 2:
                phones.append("[")
        return phones

    @staticmethod
    def _chinese_g2p(text: str) -> List[str]:
        """Chinese G2P using pypinyin with correct decomposition."""
        phonemes = []
        pinyins = pypinyin.pinyin(text, style=pypinyin.Style.TONE3, neutral_tone_with_five=True)
        
        for p_list in pinyins:
            syllable_with_tone = p_list[0]
            
            syllable = syllable_with_tone.rstrip('12345')
            tone = syllable_with_tone[len(syllable):]

            if not tone:
                tone = '5'

            initial = get_initials(syllable, strict=False)
            final = get_finals(syllable, strict=False)

            if initial:
                phonemes.append(initial)
            if final:
                phonemes.append(f"{final}{tone}")
        
        return phonemes

    @staticmethod
    def g2p(text: str, with_prosody: bool = True) -> List[str]:
        if not text.strip():
            return []

        if is_chinese(text):
            return JapaneseG2P._chinese_g2p(text)

        # Japanese processing
        norm_text = JapaneseG2P._text_normalize(text)
        japanese_segments = _JAPANESE_MARKS_RE.split(norm_text)
        punctuation_marks = _JAPANESE_MARKS_RE.findall(norm_text)
        phonemes = []
        for i, segment in enumerate(japanese_segments):
            if segment:
                if with_prosody:
                    phones = JapaneseG2P._pyopenjtalk_g2p_prosody(segment)[1:-1]
                else:
                    phones = pyopenjtalk.g2p(segment).split(" ")
                phonemes.extend(phones)
            if i < len(punctuation_marks):
                mark = punctuation_marks[i].strip()
                if mark:
                    phonemes.append(mark)
        processed_phonemes = [JapaneseG2P._post_replace_phoneme(p) for p in phonemes]
        return processed_phonemes

def japanese_to_phones(text: str) -> list[int]:
    phones = JapaneseG2P.g2p(text)
    # Replace any phoneme not in the symbol list with UNK
    valid_phones = []
    for ph in phones:
        if ph in symbol_to_id_v2:
            valid_phones.append(ph)
        else:
            # This is a fallback for safety, ideally all phonemes should be in the list
            valid_phones.append("UNK")
            
    ids = [symbol_to_id_v2[ph] for ph in valid_phones]
    return ids
