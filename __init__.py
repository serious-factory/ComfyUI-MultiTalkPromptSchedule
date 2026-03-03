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

# Prevent ComfyUI from caching nodes whose outputs get mutated
# in-place by multitalk_loop during sampling. Without this, the
# second generation receives corrupted cached data → frozen video.
def _disable_cache_for_mutable_nodes():
    import sys, logging
    _log = logging.getLogger("ComfyUI-MultiTalkPromptSchedule")

    # Scan sys.modules to find the classes regardless of module naming
    target_classes = ["MultiTalkWav2VecEmbeds", "WanVideoImageToVideoMultiTalk", "WanVideoModelLoader"]
    for mod_key, mod in list(sys.modules.items()):
        if mod is None:
            continue
        for cls_name in target_classes:
            cls = getattr(mod, cls_name, None)
            if cls is not None and isinstance(cls, type) and not hasattr(cls, '_cache_disabled'):
                cls.IS_CHANGED = classmethod(lambda cls, **kw: float("nan"))
                cls._cache_disabled = True
                _log.info(f"[MultiTalkPromptSchedule] Disabled cache for {cls_name} (in {mod_key})")

_disable_cache_for_mutable_nodes()

NODE_CLASS_MAPPINGS = {
    "MultiTalkPromptSchedule": MultiTalkPromptSchedule,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiTalkPromptSchedule": "MultiTalk Prompt Schedule",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
