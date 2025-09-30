from ..Utils.Utils import LRUCacheDict
from ..Japanese.JapaneseG2P import japanese_to_phones
from ..Utils.Constants import BERT_FEATURE_DIM
from ..Audio.Audio import load_audio
from ..ModelManager import model_manager

import os
import numpy as np
import soxr
from typing import Optional


class ReferenceAudio:
    _prompt_cache: dict[str, 'ReferenceAudio'] = LRUCacheDict(
        capacity=int(os.getenv('Max_Cached_Reference_Audio', '10')))

    def __new__(cls, prompt_wav: str, prompt_text: str, language: str = "ja"):
        if prompt_wav in cls._prompt_cache:
            instance = cls._prompt_cache[prompt_wav]
            if instance.text != prompt_text or getattr(instance, 'language', None) != language:
                instance.set_text(prompt_text, language)
            return instance

        instance = super().__new__(cls)
        cls._prompt_cache[prompt_wav] = instance
        return instance

    def __init__(self, prompt_wav: str, prompt_text: str, language: str = "ja"):
        if hasattr(self, '_initialized'):
            return

        # 文本相关。
        self.text: str = prompt_text
        self.phonemes_seq: Optional[np.ndarray] = None
        self.text_bert: Optional[np.ndarray] = None
        self.language: str = language
        self.set_text(prompt_text, language)

        # 音频相关。
        self.audio_32k: Optional[np.ndarray] = load_audio(
            audio_path=prompt_wav,
            target_sampling_rate=32000
        )
        audio_16k: np.ndarray = soxr.resample(self.audio_32k, 32000, 16000, quality='hq')
        audio_16k = np.expand_dims(audio_16k, axis=0)  # 增加 Batch_Size 维度

        if not model_manager.cn_hubert:
            model_manager.load_cn_hubert()
        self.ssl_content: Optional[np.ndarray] = model_manager.cn_hubert.run(
            None, {'input_values': audio_16k}
        )[0]

        self._initialized = True

    def set_text(self, prompt_text: str, language: str = "ja") -> None:
        self.text = prompt_text
        self.language = language
        if language == "zh":
            from ..Chinese.ChineseG2P import chinese_to_phones
            self.phonemes_seq = np.array([chinese_to_phones(prompt_text)], dtype=np.int64)
            # BERT 特征提取
            self.text_bert = self._extract_bert_features(prompt_text)
        else:
            self.phonemes_seq = np.array([japanese_to_phones(prompt_text)], dtype=np.int64)
            self.text_bert = np.zeros((self.phonemes_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)

    def _extract_bert_features(self, text: str) -> np.ndarray:
        """
        使用官方 Genie BERT/roberta ONNX 模型提取中文文本的语义特征。
        """
        import onnxruntime as ort
        import os
        # 优先使用官方 Genie 资产路径
        mika_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Models", "Mika", "character_model", "misono_mika")
        official_onnx = os.path.join(mika_dir, "bert.onnx")
        official_tokenizer = mika_dir  # 假设 tokenizer 文件与 onnx 同目录
        # 如果官方 bert.onnx 不存在，则 fallback 到自定义 roberta.onnx
        if os.path.exists(official_onnx):
            onnx_path = official_onnx
            tokenizer_path = official_tokenizer
        else:
            roberta_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Models", "Roberta")
            onnx_path = os.path.join(roberta_dir, "roberta.onnx")
            tokenizer_path = roberta_dir
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=128)
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        outputs = session.run(None, {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]})
        last_hidden_state = outputs[0]  # shape: (batch, seq_len, 1024)
        features = last_hidden_state[0]
        target_len = self.phonemes_seq.shape[1]
        if features.shape[0] >= target_len:
            features = features[:target_len]
        else:
            pad = np.zeros((target_len - features.shape[0], BERT_FEATURE_DIM), dtype=np.float32)
            features = np.concatenate([features, pad], axis=0)
        return features.astype(np.float32)

    @classmethod
    def clear_cache(cls) -> None:
        """清空 ReferenceAudio 的缓存"""
        cls._prompt_cache.clear()
