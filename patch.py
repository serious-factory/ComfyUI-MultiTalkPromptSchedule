"""
Monkey-patch for multitalk_loop.py prompt selection.

Replaces the original prompt selection block (iteration-based) with a
frame-based prompt_schedule-aware version. Falls back to original behavior
when no prompt_schedule is present in text_embeds.
"""

import logging
import textwrap

log = logging.getLogger("ComfyUI-MultiTalkPromptSchedule")

_PATCHED = False


def _select_prompt(text_embeds, audio_start_idx, iteration_count, log_fn):
    """Frame-based prompt selection with fallback to iteration-based.

    Returns:
        positive: list of prompt embeddings for this window
    """
    prompt_schedule = text_embeds.get("prompt_schedule", None)
    nag_schedule = text_embeds.get("nag_schedule", None)

    if prompt_schedule is not None:
        # Frame-based prompt selection (MultiTalkPromptSchedule node)
        prompt_index = len(prompt_schedule) - 1  # default to last
        for idx, (sched_start, sched_end) in enumerate(prompt_schedule):
            if audio_start_idx < sched_end:
                prompt_index = idx
                break
        positive = [text_embeds["prompt_embeds"][prompt_index]]

        # Per-scene NAG negative if available
        if nag_schedule is not None and prompt_index in nag_schedule:
            nag_index = nag_schedule[prompt_index]
            nag_embed = [text_embeds["all_nag_embeds"][nag_index]]
            text_embeds["nag_prompt_embeds"] = nag_embed
            text_embeds["negative_prompt_embeds"] = nag_embed
            log_fn(f"Using per-scene NAG negative for scene {prompt_index}")
        elif nag_schedule is not None:
            if "all_nag_embeds" in text_embeds:
                text_embeds["nag_prompt_embeds"] = [text_embeds["all_nag_embeds"][0]]
                text_embeds["negative_prompt_embeds"] = [text_embeds["all_nag_embeds"][0]]

        sched_start, sched_end = prompt_schedule[prompt_index]
        log_fn(
            f"Using scheduled prompt {prompt_index} "
            f"(frames {sched_start}-{sched_end}) "
            f"at audio_start_idx={audio_start_idx}"
        )
    elif len(text_embeds["prompt_embeds"]) > 1:
        # Fallback: iteration-based (original Kijai behavior, pipe separator)
        prompt_index = min(
            iteration_count, len(text_embeds["prompt_embeds"]) - 1
        )
        positive = [text_embeds["prompt_embeds"][prompt_index]]
        log_fn(f"Using prompt index: {prompt_index}")
    else:
        positive = text_embeds["prompt_embeds"]

    return positive


def apply_prompt_schedule_patch():
    """Monkey-patch multitalk_loop to support prompt_schedule in text_embeds."""
    global _PATCHED
    if _PATCHED:
        return

    import importlib

    loop_mod = None
    # Try both naming conventions (hyphen vs underscore)
    for module_name in [
        "custom_nodes.ComfyUI-WanVideoWrapper.multitalk.multitalk_loop",
        "custom_nodes.ComfyUI_WanVideoWrapper.multitalk.multitalk_loop",
    ]:
        try:
            loop_mod = importlib.import_module(module_name)
            break
        except ImportError:
            continue

    if loop_mod is None:
        log.warning(
            "[MultiTalkPromptSchedule] ComfyUI-WanVideoWrapper not found. "
            "Patch not applied — the node will still load but scheduling "
            "won't work until WanVideoWrapper is installed."
        )
        return

    import inspect
    import re

    original_fn = loop_mod.multitalk_loop

    # Get the source to check if the patch is already present
    # (e.g. user already manually patched)
    try:
        src = inspect.getsource(original_fn)
        if "prompt_schedule" in src:
            log.info(
                "[MultiTalkPromptSchedule] multitalk_loop already contains "
                "prompt_schedule logic. Skipping monkey-patch."
            )
            _PATCHED = True
            return
    except (OSError, TypeError):
        # Can't inspect source (compiled, etc.) — proceed with patch
        pass

    # ── Build the patched function ──────────────────────────────────────
    # We wrap the original function and intercept the prompt selection.
    # The trick: multitalk_loop is a method (self, **kwargs).  We wrap it
    # so that before it runs, we inject a _select_prompt callback into
    # text_embeds that the original code path will pick up.
    #
    # Strategy: We pre-process text_embeds so that when the original code
    # hits `if len(text_embeds["prompt_embeds"]) > 1:`, the embeds are
    # already set to the correct prompt for the FIRST window.  For
    # subsequent windows we can't intercept since the loop is internal.
    #
    # Better strategy: We actually patch the source code of the function
    # at the AST/text level to inject our prompt_schedule block.

    src = inspect.getsource(original_fn)

    # The original prompt selection block we need to replace:
    original_block = textwrap.dedent("""\
        # Use the appropriate prompt for this section
        if len(text_embeds["prompt_embeds"]) > 1:
            prompt_index = min(iteration_count, len(text_embeds["prompt_embeds"]) - 1)
            positive = [text_embeds["prompt_embeds"][prompt_index]]
            log.info(f"Using prompt index: {prompt_index}")
        else:
            positive = text_embeds["prompt_embeds"]""")

    patched_block = textwrap.dedent("""\
        # Use the appropriate prompt for this section
        # [Patched by ComfyUI-MultiTalkPromptSchedule]
        prompt_schedule = text_embeds.get("prompt_schedule", None)
        nag_schedule = text_embeds.get("nag_schedule", None)
        if prompt_schedule is not None:
            prompt_index = len(prompt_schedule) - 1
            for idx, (sched_start, sched_end) in enumerate(prompt_schedule):
                if audio_start_idx < sched_end:
                    prompt_index = idx
                    break
            positive = [text_embeds["prompt_embeds"][prompt_index]]
            if nag_schedule is not None and prompt_index in nag_schedule:
                nag_index = nag_schedule[prompt_index]
                nag_embed = [text_embeds["all_nag_embeds"][nag_index]]
                text_embeds["nag_prompt_embeds"] = nag_embed
                text_embeds["negative_prompt_embeds"] = nag_embed
                log.info(f"Using per-scene NAG negative for scene {prompt_index}")
            elif nag_schedule is not None:
                if "all_nag_embeds" in text_embeds:
                    text_embeds["nag_prompt_embeds"] = [text_embeds["all_nag_embeds"][0]]
                    text_embeds["negative_prompt_embeds"] = [text_embeds["all_nag_embeds"][0]]
            sched_start, sched_end = prompt_schedule[prompt_index]
            log.info(f"Using scheduled prompt {prompt_index} (frames {sched_start}-{sched_end}) at audio_start_idx={audio_start_idx}")
        elif len(text_embeds["prompt_embeds"]) > 1:
            prompt_index = min(iteration_count, len(text_embeds["prompt_embeds"]) - 1)
            positive = [text_embeds["prompt_embeds"][prompt_index]]
            log.info(f"Using prompt index: {prompt_index}")
        else:
            positive = text_embeds["prompt_embeds"]""")

    # Normalize whitespace for matching: the source may have different
    # indentation (8 spaces inside the while loop).
    def _normalize_indent(block):
        """Strip common leading whitespace so we can match flexibly."""
        lines = block.splitlines()
        return [l.strip() for l in lines if l.strip()]

    original_lines = _normalize_indent(original_block)

    # Find the block in the source by matching stripped lines
    src_lines = src.splitlines()
    match_start = None
    for i in range(len(src_lines)):
        if src_lines[i].strip() == original_lines[0]:
            # Check if subsequent lines match
            matched = True
            for j, orig_line in enumerate(original_lines):
                if i + j >= len(src_lines):
                    matched = False
                    break
                if src_lines[i + j].strip() != orig_line:
                    matched = False
                    break
            if matched:
                match_start = i
                break

    if match_start is None:
        log.warning(
            "[MultiTalkPromptSchedule] Could not find the original prompt "
            "selection block in multitalk_loop source. The upstream code may "
            "have changed. Patch not applied."
        )
        return

    # Determine the indentation of the matched block
    indent = ""
    first_line = src_lines[match_start]
    indent = first_line[: len(first_line) - len(first_line.lstrip())]

    # Build the replacement with proper indentation
    patched_lines = []
    for line in patched_block.splitlines():
        if line.strip():
            patched_lines.append(indent + line)
        else:
            patched_lines.append("")

    # Replace in source
    match_end = match_start + len(original_lines)
    new_src_lines = (
        src_lines[:match_start] + patched_lines + src_lines[match_end:]
    )
    new_src = "\n".join(new_src_lines)

    # Remove the `def multitalk_loop(self, **kwargs):` indentation
    # since we'll exec it as a module-level function.
    # Actually, we need to dedent the function definition first.
    new_src = textwrap.dedent(new_src)

    # Compile and exec in the module's namespace
    try:
        code = compile(new_src, loop_mod.__file__, "exec")
        exec(code, loop_mod.__dict__)
        log.info(
            "[MultiTalkPromptSchedule] Successfully patched multitalk_loop "
            "with prompt_schedule support."
        )
        _PATCHED = True

        # CRITICAL: nodes_sampler.py imports multitalk_loop with
        # `from .multitalk.multitalk_loop import multitalk_loop`
        # which creates a separate reference. We must update it too,
        # otherwise the sampler keeps calling the original function.
        for sampler_module_name in [
            "custom_nodes.ComfyUI-WanVideoWrapper.nodes_sampler",
            "custom_nodes.ComfyUI_WanVideoWrapper.nodes_sampler",
        ]:
            try:
                sampler_mod = importlib.import_module(sampler_module_name)
                sampler_mod.multitalk_loop = loop_mod.multitalk_loop
                log.info(
                    "[MultiTalkPromptSchedule] Also patched reference in "
                    "nodes_sampler module."
                )
                break
            except (ImportError, AttributeError):
                continue

    except Exception as e:
        log.error(
            f"[MultiTalkPromptSchedule] Failed to compile patched "
            f"multitalk_loop: {e}. Patch not applied."
        )
