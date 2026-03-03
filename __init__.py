"""
ComfyUI-MultiTalkPromptSchedule
Standalone custom node that adds frame-based prompt scheduling to InfiniteTalk/MultiTalk.

Works by monkey-patching the prompt selection logic in WanVideoWrapper's multitalk_loop
at import time. No files in ComfyUI-WanVideoWrapper are modified.
"""

from .prompt_schedule import MultiTalkPromptSchedule
from .patch import apply_prompt_schedule_patch

# Apply the monkey-patch on import
apply_prompt_schedule_patch()

NODE_CLASS_MAPPINGS = {
    "MultiTalkPromptSchedule": MultiTalkPromptSchedule,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiTalkPromptSchedule": "MultiTalk Prompt Schedule",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
