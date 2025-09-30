from rich.console import Console
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..Audio.ReferenceAudio import ReferenceAudio

console: Console = Console()


class Context:
    def __init__(self):
        self.current_speaker: str = ''
        self.current_prompt_audio: Optional['ReferenceAudio'] = None
        self.language: str = 'ja'  # 默认日语，可切换为 zh


context: Context = Context()
