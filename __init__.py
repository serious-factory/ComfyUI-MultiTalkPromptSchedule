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
    import importlib, logging
    _log = logging.getLogger("ComfyUI-MultiTalkPromptSchedule")
    for mod_name in [
        "custom_nodes.ComfyUI-WanVideoWrapper.nodes",
        "custom_nodes.ComfyUI_WanVideoWrapper.nodes",
    ]:
        try:
            mod = importlib.import_module(mod_name)
            break
        except ImportError:
            continue
    else:
        return

    # MultiTalkWav2VecEmbeds: audio_embedding gets padded in-place
    # WanVideoImageToVideoMultiTalk: image_embeds dict gets mutated
    for cls_name in ["MultiTalkWav2VecEmbeds", "WanVideoImageToVideoMultiTalk"]:
        cls = getattr(mod, cls_name, None)
        if cls is not None and not hasattr(cls, '_orig_IS_CHANGED'):
            cls.IS_CHANGED = classmethod(lambda cls, **kw: float("nan"))
            _log.info(f"[MultiTalkPromptSchedule] Disabled cache for {cls_name}")

_disable_cache_for_mutable_nodes()

NODE_CLASS_MAPPINGS = {
    "MultiTalkPromptSchedule": MultiTalkPromptSchedule,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiTalkPromptSchedule": "MultiTalk Prompt Schedule",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
