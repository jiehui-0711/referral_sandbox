"""被測 Agent。這裡才是受測對象，sim/ 那邊是世界裡的人。
目前只有一個被測角色：轉診協調員。
它是整場唯一會主動發起行動的角色。

    backends.py  模型後端：怎麼跟一個模型講話
                 （本機的 Ollama、OpenAI 相容的服務、錄播）
    scripted.py  照腳本執行的假 Agent
    react.py     最單純的 ReAct 迴圈，刻意不做任何提示工程
    schema.py    把工具定義翻譯成模型看得懂的格式
"""
from .backends import (Backend, BackendError, OllamaBackend, OpenAICompatBackend,
                       ReplayBackend, build_backend)

__all__ = ["Backend", "BackendError", "OllamaBackend", "OpenAICompatBackend",
           "ReplayBackend", "build_backend"]
