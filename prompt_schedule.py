class MultiTalkPromptSchedule:
    """Assigns prompts to specific frame ranges for scene direction in MultiTalk.

    Format: one line per scene, each line is "frames: prompt text"
    Optional per-scene NAG negative with "|||": "frames: positive ||| negative"
    Lines without "|||" use the global negative_prompt field for NAG.

    The per-scene negatives feed NAG (Normalized Attention Guidance), which
    operates inside cross-attention layers. This is the only negative guidance
    mechanism active at cfg=1 (standard InfiniteTalk setting).

    Example:
        120: Person 1 speaks to camera ||| looking sideways, looking away
        120: Person 2 speaks to person 1 ||| looking at camera, wandering eyes
        120: Both look at camera and smile
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "t5": ("WANTEXTENCODER",),
                "schedule": ("STRING", {
                    "default": (
                        "120: Person 1 speaks to the camera, person 2 listens\n"
                        "120: Person 2 responds, person 1 nods\n"
                        "120: Both look at the camera and smile"
                    ),
                    "multiline": True,
                    "tooltip": (
                        "One line per scene. Format: frames: prompt text "
                        "||| optional NAG negative prompt"
                    ),
                }),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Default negative prompt for NAG guidance. "
                        "Used for all scenes without a per-scene negative (|||). "
                        "Also used as the negative_text_embeds output for "
                        "WanVideoApplyNAG."
                    ),
                }),
            },
            "optional": {
                "force_offload": ("BOOLEAN", {"default": True}),
                "device": (["gpu", "cpu"], {"default": "gpu"}),
            },
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", "WANVIDEOTEXTEMBEDS")
    RETURN_NAMES = ("text_embeds", "negative_text_embeds")
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-execution every time — prevents ComfyUI from serving
        # a cached text_embeds dict that was mutated in-place by the
        # sampler's multitalk_loop during the previous generation.
        return float("nan")
    DESCRIPTION = """Scene director for MultiTalk: assign prompts to frame ranges.

Format: one line per scene
  <frames>: <prompt>
  <frames>: <prompt> ||| <NAG negative prompt>

Lines without ||| use the global negative_prompt field.
Per-scene negatives feed NAG (active at cfg=1).

Connect negative_text_embeds to WanVideoApplyNAG → nag_text_embeds.

Example:
  120: Person 1 speaks to camera ||| looking sideways, turned away
  120: Person 2 responds, person 1 nods
  120: Both look at the camera and smile

Each number is the duration in frames (120 = 5s at 24fps).
"""

    @staticmethod
    def parse_schedule(text):
        """Parse schedule text into list of (frames, prompt, neg_or_None)."""
        entries = []
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            colon_idx = line.find(":")
            if colon_idx == -1:
                raise ValueError(f"Invalid line (missing ':'): {line}")
            frames_str = line[:colon_idx].strip()
            rest = line[colon_idx + 1 :].strip()
            try:
                frames = int(frames_str)
            except ValueError:
                raise ValueError(
                    f"Invalid frame count '{frames_str}' in line: {line}"
                )
            if frames <= 0:
                raise ValueError(
                    f"Frame count must be > 0, got {frames} in line: {line}"
                )
            if not rest:
                raise ValueError(f"Empty prompt in line: {line}")

            if "|||" in rest:
                parts = rest.split("|||", 1)
                prompt = parts[0].strip()
                neg_prompt = parts[1].strip()
            else:
                prompt = rest
                neg_prompt = None

            entries.append((frames, prompt, neg_prompt))
        return entries

    def process(
        self, t5, schedule, negative_prompt="", force_offload=True, device="gpu"
    ):
        import gc

        import torch
        from comfy import model_management as mm
        from tqdm import tqdm

        # Import from WanVideoWrapper (our dependency)
        import importlib

        log = None
        set_module_tensor_to_device = None
        for mod_name in [
            "custom_nodes.ComfyUI-WanVideoWrapper.utils",
            "custom_nodes.ComfyUI_WanVideoWrapper.utils",
        ]:
            try:
                utils_mod = importlib.import_module(mod_name)
                log = utils_mod.log
                set_module_tensor_to_device = utils_mod.set_module_tensor_to_device
                break
            except ImportError:
                continue
        if log is None:
            raise ImportError(
                "ComfyUI-WanVideoWrapper not found. "
                "Please install it first."
            )

        # Parse the schedule text
        entries = self.parse_schedule(schedule)
        if not entries:
            raise ValueError("No valid entries found in schedule")

        prompts = []
        frame_schedule = []
        per_scene_negatives = {}
        current_frame = 0

        for i, (frames, prompt, neg_prompt) in enumerate(entries):
            prompts.append(prompt)
            frame_schedule.append((current_frame, current_frame + frames))
            if neg_prompt is not None:
                per_scene_negatives[i] = neg_prompt
            current_frame += frames

        log.info(f"[MultiTalkPromptSchedule] {len(prompts)} prompts:")
        for i, (prompt, (start, end)) in enumerate(
            zip(prompts, frame_schedule)
        ):
            neg_info = ""
            if i in per_scene_negatives:
                neg_info = f' [NAG negative: "{per_scene_negatives[i][:40]}"]'
            log.info(
                f"  Prompt {i+1}: frames {start}-{end} "
                f"({end-start} frames){neg_info} = "
                f'"{prompt[:80]}"'
            )

        # Encode all prompts
        offload_device = mm.unet_offload_device()
        if device == "gpu":
            device_to = mm.get_torch_device()
        else:
            device_to = torch.device("cpu")

        encoder = t5["model"]
        dtype = t5["dtype"]

        if encoder.quantization == "fp8_e4m3fn":
            cast_dtype = torch.float8_e4m3fn
        else:
            cast_dtype = encoder.dtype

        params_to_keep = {"norm", "pos_embedding", "token_embedding"}
        if hasattr(encoder, "state_dict"):
            model_state_dict = encoder.state_dict
        else:
            model_state_dict = encoder.model.state_dict()

        params_list = list(encoder.model.named_parameters())
        pbar = tqdm(params_list, desc="Loading T5 parameters", leave=True)
        for name, param in pbar:
            dtype_to_use = (
                dtype
                if any(keyword in name for keyword in params_to_keep)
                else cast_dtype
            )
            value = model_state_dict[name]
            set_module_tensor_to_device(
                encoder.model, name, device=device_to,
                dtype=dtype_to_use, value=value,
            )
        del model_state_dict
        if hasattr(encoder, "state_dict"):
            del encoder.state_dict
            mm.soft_empty_cache()
            gc.collect()

        # Collect unique negative prompts to encode.
        # These serve as NAG negatives (the only negative guidance at cfg=1).
        unique_negatives = [negative_prompt]  # index 0 = global default
        neg_text_to_index = {negative_prompt: 0}
        nag_schedule = {}

        for scene_idx, neg_text in per_scene_negatives.items():
            if neg_text not in neg_text_to_index:
                neg_text_to_index[neg_text] = len(unique_negatives)
                unique_negatives.append(neg_text)
            nag_schedule[scene_idx] = neg_text_to_index[neg_text]

        log.info(
            f"[MultiTalkPromptSchedule] Encoding "
            f"{len(unique_negatives)} unique negative prompt(s) for NAG"
        )

        with torch.autocast(
            device_type=mm.get_autocast_device(device_to),
            dtype=encoder.dtype,
            enabled=encoder.quantization != "disabled",
        ):
            context = encoder(prompts, device_to)
            context_null = encoder(unique_negatives, device_to)

        if force_offload:
            for name, param in encoder.model.named_parameters():
                set_module_tensor_to_device(
                    encoder.model, name, device=offload_device,
                    dtype=param.dtype,
                    value=torch.zeros_like(param, device=offload_device),
                )
            mm.soft_empty_cache()
            gc.collect()

        text_embeds = {
            "prompt_embeds": context,
            "negative_prompt_embeds": [context_null[0]],
            "prompt_schedule": frame_schedule,
        }

        # Per-scene NAG negatives: stored in text_embeds so the
        # multitalk_loop patch can swap nag_prompt_embeds per scene.
        if nag_schedule:
            text_embeds["all_nag_embeds"] = context_null
            text_embeds["nag_schedule"] = nag_schedule

        # negative_text_embeds output: connect to WanVideoApplyNAG → nag_text_embeds.
        # This sets the global/default NAG negative embedding.
        negative_text_embeds = {
            "prompt_embeds": [context_null[0]],
        }

        log.info(
            f"[MultiTalkPromptSchedule] Encoded {len(context)} prompts, "
            f"schedule: {frame_schedule}"
        )
        if nag_schedule:
            log.info(
                f"[MultiTalkPromptSchedule] Per-scene NAG negatives: "
                f"{nag_schedule}"
            )

        return (text_embeds, negative_text_embeds)
