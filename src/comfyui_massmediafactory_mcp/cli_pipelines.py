"""
Pre-tested pipeline definitions for mmf CLI.

Each pipeline encodes exact parameters that produce correct results for a specific
multi-stage workflow. Parameters are hardcoded from production testing — do NOT
make them configurable unless you've verified the new values produce good output.
"""

import json
import random


def run_pipeline(name: str, args) -> dict:
    """Dispatch to the named pipeline."""
    pipelines = {
        "i2v": _pipeline_i2v,
        "upscale": _pipeline_upscale,
        "viral-short": _pipeline_viral_short,
        "t2v-styled": _pipeline_t2v_styled,
        "bio-to-video": _pipeline_bio_to_video,
    }
    fn = pipelines.get(name)
    if fn is None:
        return {
            "error": f"Unknown pipeline: {name}. Available: {', '.join(sorted(pipelines))}",
            "code": "INVALID_PARAMS",
        }
    return fn(args)


def run_telestyle(mode: str, args) -> dict:
    """Run TeleStyle image or video transfer."""
    from . import execution, templates as tmpl

    seed = args.seed if args.seed is not None else random.randint(1, 2**32 - 1)

    if mode == "image":
        # Upload content and style images
        content_upload = execution.upload_image(args.content)
        if "error" in content_upload:
            return content_upload
        style_upload = execution.upload_image(args.style)
        if "error" in style_upload:
            return style_upload

        template = tmpl.load_template("telestyle_image")
        if "error" in template:
            return template

        params = {
            "CONTENT_IMAGE": content_upload["name"],
            "STYLE_IMAGE": style_upload["name"],
            "SEED": seed,
            "CFG": args.cfg or 2.0,  # Tested: 2.0 is optimal for TeleStyle
            "STEPS": args.steps or 20,
        }
        wf = tmpl.inject_parameters(template, params)

    elif mode == "video":
        # Upload style image and video
        style_upload = execution.upload_image(args.style)
        if "error" in style_upload:
            return style_upload

        template = tmpl.load_template("telestyle_video")
        if "error" in template:
            return template

        params = {
            "VIDEO_PATH": args.content,  # Video path used directly
            "STYLE_PATH": style_upload["name"],
            "SEED": seed,
            "CFG": args.cfg or 1.0,  # Tested: 1.0 for video style transfer
            "STEPS": args.steps or 12,
            "FPS": 24,
        }
        wf = tmpl.inject_parameters(template, params)
    else:
        return {"error": f"Unknown telestyle mode: {mode}", "code": "INVALID_PARAMS"}

    # Execute and wait
    exec_result = execution.execute_workflow(wf)
    if "error" in exec_result:
        return exec_result

    timeout = args.timeout or 600
    output = execution.wait_for_completion(exec_result["prompt_id"], timeout_seconds=timeout, workflow=wf)

    # Auto-download if --output specified
    if getattr(args, "output", None) and output.get("outputs"):
        asset_id = output["outputs"][0].get("asset_id")
        if asset_id:
            dl = execution.download_output(asset_id, args.output)
            output["downloaded"] = dl

    return output


def _pipeline_i2v(args) -> dict:
    """Image-to-Video pipeline: upload → I2V → download."""
    from . import execution, templates as tmpl

    if not args.image:
        return {"error": "--image required for i2v pipeline", "code": "INVALID_PARAMS"}
    if not args.prompt:
        return {"error": "--prompt required for i2v pipeline", "code": "INVALID_PARAMS"}

    model = args.model or "wan"
    seed = args.seed if args.seed is not None else random.randint(1, 2**32 - 1)

    # Upload image
    upload = execution.upload_image(args.image)
    if "error" in upload:
        return upload

    # Select template based on model
    template_map = {
        "wan": "wan26_img2vid",
        "ltx": "ltx2_img2vid",
        "hunyuan": "hunyuan15_img2vid",
    }
    template_name = template_map.get(model)
    if not template_name:
        return {
            "error": f"No i2v template for model: {model}. Available: {', '.join(sorted(template_map))}",
            "code": "INVALID_PARAMS",
        }

    template = tmpl.load_template(template_name)
    if "error" in template:
        return template

    # Model-specific tested defaults
    if model == "wan":
        params = {
            "IMAGE_PATH": upload["name"],
            "PROMPT": args.prompt,
            "NEGATIVE": "blurry, low quality, distorted, watermark",
            "SEED": seed,
            "CFG": 5.0,
            "SHIFT": 5.0,
            "STEPS": 30,
            "FRAMES": 81,
            "WIDTH": 832,
            "HEIGHT": 480,
            "FPS": 16,
            "NOISE_AUG": 0.0,
        }
    elif model == "ltx":
        params = {
            "IMAGE_PATH": upload["name"],
            "PROMPT": args.prompt,
            "NEGATIVE": "worst quality, blurry",
            "SEED": seed,
            "STEPS": 30,
            "FRAMES": 97,
            "WIDTH": 768,
            "HEIGHT": 512,
            "FPS": 24,
        }
    elif model == "hunyuan":
        params = {
            "IMAGE_PATH": upload["name"],
            "PROMPT": args.prompt,
            "NEGATIVE": "blurry, low quality",
            "SEED": seed,
            "STEPS": 30,
            "FRAMES": 81,
            "WIDTH": 848,
            "HEIGHT": 480,
            "FPS": 24,
        }

    wf = tmpl.inject_parameters(template, params)

    exec_result = execution.execute_workflow(wf)
    if "error" in exec_result:
        return exec_result

    timeout = args.timeout or 600
    output = execution.wait_for_completion(exec_result["prompt_id"], timeout_seconds=timeout, workflow=wf)

    if getattr(args, "output", None) and output.get("outputs"):
        asset_id = output["outputs"][0].get("asset_id")
        if asset_id:
            dl = execution.download_output(asset_id, args.output)
            output["downloaded"] = dl

    return output


def _pipeline_upscale(args) -> dict:
    """Upscale pipeline: upload → FLUX Ultimate Upscale → download."""
    from . import execution, templates as tmpl

    if not args.image:
        return {"error": "--image required for upscale pipeline", "code": "INVALID_PARAMS"}

    seed = args.seed if args.seed is not None else random.randint(1, 2**32 - 1)
    factor = args.factor or 2.0

    upload = execution.upload_image(args.image)
    if "error" in upload:
        return upload

    template = tmpl.load_template("flux2_ultimate_upscale")
    if "error" in template:
        return template

    params = {
        "IMAGE_PATH": upload["name"],
        "SCALE_FACTOR": factor,
        "DENOISE": 0.35,  # Tested: 0.35 preserves details while enhancing
        "TILE_SIZE": 512,
        "SEED": seed,
        "STEPS": 20,
    }
    wf = tmpl.inject_parameters(template, params)

    exec_result = execution.execute_workflow(wf)
    if "error" in exec_result:
        return exec_result

    timeout = args.timeout or 600
    output = execution.wait_for_completion(exec_result["prompt_id"], timeout_seconds=timeout, workflow=wf)

    if getattr(args, "output", None) and output.get("outputs"):
        asset_id = output["outputs"][0].get("asset_id")
        if asset_id:
            dl = execution.download_output(asset_id, args.output)
            output["downloaded"] = dl

    return output


def _pipeline_viral_short(args) -> dict:
    """
    KDH viral short pipeline:
    Stage 1: Qwen T2I (character keyframe)
    Stage 2: TeleStyle Image (style transfer)
    Stage 3: WAN I2V (animate to video)
    """
    from . import execution, templates as tmpl

    if not args.prompt:
        return {"error": "--prompt required for viral-short pipeline", "code": "INVALID_PARAMS"}
    if not args.style_image:
        return {"error": "--style-image required for viral-short pipeline", "code": "INVALID_PARAMS"}

    seed = args.seed if args.seed is not None else random.randint(1, 2**32 - 1)

    # Stage 1: Generate character keyframe with Qwen
    t2i_template = tmpl.load_template("qwen_txt2img")
    if "error" in t2i_template:
        return t2i_template

    t2i_params = {
        "PROMPT": args.prompt,
        "NEGATIVE": "blurry, low quality, text, watermark, ugly, deformed",
        "SEED": seed,
        "WIDTH": 832,
        "HEIGHT": 1216,  # Portrait for characters
        "SHIFT": 7.0,  # Tested: 7.0 for Qwen T2I
    }
    t2i_wf = tmpl.inject_parameters(t2i_template, t2i_params)

    t2i_exec = execution.execute_workflow(t2i_wf)
    if "error" in t2i_exec:
        return {"error": f"Stage 1 (T2I) failed: {t2i_exec['error']}", "code": "PIPELINE_ERROR", "stage": "t2i"}

    t2i_output = execution.wait_for_completion(t2i_exec["prompt_id"], timeout_seconds=300, workflow=t2i_wf)
    if t2i_output.get("status") != "completed" or not t2i_output.get("outputs"):
        return {
            "error": f"Stage 1 (T2I) failed: {t2i_output.get('status')}",
            "code": "PIPELINE_ERROR",
            "stage": "t2i",
            "details": t2i_output,
        }

    keyframe_asset_id = t2i_output["outputs"][0].get("asset_id")

    # Stage 2: TeleStyle the keyframe
    style_upload = execution.upload_image(args.style_image)
    if "error" in style_upload:
        return style_upload

    # Download keyframe to upload as content for TeleStyle
    from .assets import get_registry

    registry = get_registry()
    keyframe_asset = registry.get_asset(keyframe_asset_id)
    if not keyframe_asset:
        return {"error": "Keyframe asset not found", "code": "PIPELINE_ERROR", "stage": "telestyle"}

    ts_template = tmpl.load_template("telestyle_image")
    if "error" in ts_template:
        return ts_template

    ts_params = {
        "CONTENT_IMAGE": keyframe_asset.filename,
        "STYLE_IMAGE": style_upload["name"],
        "SEED": seed,
        "CFG": 2.0,  # CRITICAL: TeleStyle CFG must be 2.0-2.5
        "STEPS": 20,
    }
    ts_wf = tmpl.inject_parameters(ts_template, ts_params)

    ts_exec = execution.execute_workflow(ts_wf)
    if "error" in ts_exec:
        return {
            "error": f"Stage 2 (TeleStyle) failed: {ts_exec['error']}",
            "code": "PIPELINE_ERROR",
            "stage": "telestyle",
        }

    ts_output = execution.wait_for_completion(ts_exec["prompt_id"], timeout_seconds=300, workflow=ts_wf)
    if ts_output.get("status") != "completed" or not ts_output.get("outputs"):
        return {
            "error": f"Stage 2 (TeleStyle) failed: {ts_output.get('status')}",
            "code": "PIPELINE_ERROR",
            "stage": "telestyle",
            "details": ts_output,
        }

    styled_asset_id = ts_output["outputs"][0].get("asset_id")
    styled_asset = registry.get_asset(styled_asset_id)
    if not styled_asset:
        return {"error": "Styled asset not found", "code": "PIPELINE_ERROR", "stage": "i2v"}

    # Stage 3: Animate styled image with WAN I2V
    i2v_template = tmpl.load_template("wan26_img2vid")
    if "error" in i2v_template:
        return i2v_template

    motion_prompt = getattr(args, "motion_prompt", None) or args.prompt
    i2v_params = {
        "IMAGE_PATH": styled_asset.filename,
        "PROMPT": motion_prompt,
        "NEGATIVE": "blurry, low quality, distorted, static, no motion",
        "SEED": seed,
        "CFG": 5.0,
        "SHIFT": 5.0,
        "STEPS": 30,
        "FRAMES": 81,
        "WIDTH": 832,
        "HEIGHT": 480,
        "FPS": 16,
        "NOISE_AUG": 0.0,
    }
    i2v_wf = tmpl.inject_parameters(i2v_template, i2v_params)

    i2v_exec = execution.execute_workflow(i2v_wf)
    if "error" in i2v_exec:
        return {"error": f"Stage 3 (I2V) failed: {i2v_exec['error']}", "code": "PIPELINE_ERROR", "stage": "i2v"}

    timeout = args.timeout or 600
    i2v_output = execution.wait_for_completion(i2v_exec["prompt_id"], timeout_seconds=timeout, workflow=i2v_wf)

    if getattr(args, "output", None) and i2v_output.get("outputs"):
        asset_id = i2v_output["outputs"][0].get("asset_id")
        if asset_id:
            dl = execution.download_output(asset_id, args.output)
            i2v_output["downloaded"] = dl

    i2v_output["pipeline"] = "viral-short"
    i2v_output["stages"] = {
        "t2i": {"asset_id": keyframe_asset_id},
        "telestyle": {"asset_id": styled_asset_id},
        "i2v": {"prompt_id": i2v_exec["prompt_id"]},
    }
    return i2v_output


def _pipeline_t2v_styled(args) -> dict:
    """
    T2V + TeleStyle pipeline:
    Stage 1: LTX T2V (generate video)
    Stage 2: TeleStyle Video (apply style)
    """
    from . import execution, templates as tmpl

    if not args.prompt:
        return {"error": "--prompt required for t2v-styled pipeline", "code": "INVALID_PARAMS"}
    if not args.style_image:
        return {"error": "--style-image required for t2v-styled pipeline", "code": "INVALID_PARAMS"}

    seed = args.seed if args.seed is not None else random.randint(1, 2**32 - 1)

    # Stage 1: Generate video with LTX
    t2v_template = tmpl.load_template("ltx2_txt2vid")
    if "error" in t2v_template:
        return t2v_template

    t2v_params = {
        "PROMPT": args.prompt,
        "NEGATIVE": "worst quality, blurry, jittery, distorted",
        "SEED": seed,
        "WIDTH": 768,
        "HEIGHT": 512,
        "FRAMES": 97,
        "FPS": 24,
    }
    t2v_wf = tmpl.inject_parameters(t2v_template, t2v_params)

    t2v_exec = execution.execute_workflow(t2v_wf)
    if "error" in t2v_exec:
        return {"error": f"Stage 1 (T2V) failed: {t2v_exec['error']}", "code": "PIPELINE_ERROR", "stage": "t2v"}

    t2v_output = execution.wait_for_completion(t2v_exec["prompt_id"], timeout_seconds=600, workflow=t2v_wf)
    if t2v_output.get("status") != "completed" or not t2v_output.get("outputs"):
        return {
            "error": f"Stage 1 (T2V) failed: {t2v_output.get('status')}",
            "code": "PIPELINE_ERROR",
            "stage": "t2v",
            "details": t2v_output,
        }

    video_asset_id = t2v_output["outputs"][0].get("asset_id")
    from .assets import get_registry

    registry = get_registry()
    video_asset = registry.get_asset(video_asset_id)
    if not video_asset:
        return {"error": "Video asset not found", "code": "PIPELINE_ERROR", "stage": "telestyle"}

    # Stage 2: TeleStyle the video
    style_upload = execution.upload_image(args.style_image)
    if "error" in style_upload:
        return style_upload

    ts_template = tmpl.load_template("telestyle_video")
    if "error" in ts_template:
        return ts_template

    ts_params = {
        "VIDEO_PATH": video_asset.filename,
        "STYLE_PATH": style_upload["name"],
        "SEED": seed,
        "CFG": 1.0,  # Tested: 1.0 for video style transfer
        "STEPS": 12,
        "FPS": 24,
    }
    ts_wf = tmpl.inject_parameters(ts_template, ts_params)

    ts_exec = execution.execute_workflow(ts_wf)
    if "error" in ts_exec:
        return {
            "error": f"Stage 2 (TeleStyle) failed: {ts_exec['error']}",
            "code": "PIPELINE_ERROR",
            "stage": "telestyle",
        }

    timeout = args.timeout or 600
    ts_output = execution.wait_for_completion(ts_exec["prompt_id"], timeout_seconds=timeout, workflow=ts_wf)

    if getattr(args, "output", None) and ts_output.get("outputs"):
        asset_id = ts_output["outputs"][0].get("asset_id")
        if asset_id:
            dl = execution.download_output(asset_id, args.output)
            ts_output["downloaded"] = dl

    ts_output["pipeline"] = "t2v-styled"
    ts_output["stages"] = {
        "t2v": {"asset_id": video_asset_id},
        "telestyle": {"prompt_id": ts_exec["prompt_id"]},
    }
    return ts_output


def _pipeline_bio_to_video(args) -> dict:
    """
    Pokedex pipeline: Bio T2I → Shiny transform → I2V
    Stage 1: Qwen T2I (generate creature from bio prompt)
    Stage 2: WAN I2V (animate)

    Note: The "shiny transform" is embedded in the prompt — the bio-prompt should
    include color/appearance descriptions. For actual color transforms, use the
    Qwen Edit template separately.
    """
    from . import execution, templates as tmpl

    bio_prompt = getattr(args, "bio_prompt", None) or args.prompt
    if not bio_prompt:
        return {"error": "--bio-prompt or --prompt required for bio-to-video pipeline", "code": "INVALID_PARAMS"}

    motion_prompt = getattr(args, "motion_prompt", None) or "the creature slowly breathes and looks around"
    seed = args.seed if args.seed is not None else random.randint(1, 2**32 - 1)

    # Incorporate shiny colors into prompt if provided
    shiny_colors = getattr(args, "shiny_colors", None)
    if shiny_colors:
        try:
            colors = json.loads(shiny_colors) if isinstance(shiny_colors, str) else shiny_colors
            color_desc = ", ".join(f"{k} color: {v}" for k, v in colors.items())
            bio_prompt = f"{bio_prompt}, with vibrant {color_desc}"
        except (json.JSONDecodeError, AttributeError):
            pass

    # Stage 1: Generate creature image
    t2i_template = tmpl.load_template("qwen_txt2img")
    if "error" in t2i_template:
        return t2i_template

    t2i_params = {
        "PROMPT": bio_prompt,
        "NEGATIVE": "blurry, low quality, text, watermark, human, person",
        "SEED": seed,
        "WIDTH": 832,
        "HEIGHT": 832,  # Square for creatures
        "SHIFT": 7.0,
    }
    t2i_wf = tmpl.inject_parameters(t2i_template, t2i_params)

    t2i_exec = execution.execute_workflow(t2i_wf)
    if "error" in t2i_exec:
        return {"error": f"Stage 1 (T2I) failed: {t2i_exec['error']}", "code": "PIPELINE_ERROR", "stage": "t2i"}

    t2i_output = execution.wait_for_completion(t2i_exec["prompt_id"], timeout_seconds=300, workflow=t2i_wf)
    if t2i_output.get("status") != "completed" or not t2i_output.get("outputs"):
        return {
            "error": f"Stage 1 (T2I) failed: {t2i_output.get('status')}",
            "code": "PIPELINE_ERROR",
            "stage": "t2i",
            "details": t2i_output,
        }

    creature_asset_id = t2i_output["outputs"][0].get("asset_id")
    from .assets import get_registry

    registry = get_registry()
    creature_asset = registry.get_asset(creature_asset_id)
    if not creature_asset:
        return {"error": "Creature image asset not found", "code": "PIPELINE_ERROR", "stage": "i2v"}

    # Stage 2: Animate with WAN I2V
    i2v_template = tmpl.load_template("wan26_img2vid")
    if "error" in i2v_template:
        return i2v_template

    i2v_params = {
        "IMAGE_PATH": creature_asset.filename,
        "PROMPT": motion_prompt,
        "NEGATIVE": "blurry, low quality, distorted, static",
        "SEED": seed,
        "CFG": 5.0,
        "SHIFT": 5.0,
        "STEPS": 30,
        "FRAMES": 81,
        "WIDTH": 832,
        "HEIGHT": 480,
        "FPS": 16,
        "NOISE_AUG": 0.0,
    }
    i2v_wf = tmpl.inject_parameters(i2v_template, i2v_params)

    i2v_exec = execution.execute_workflow(i2v_wf)
    if "error" in i2v_exec:
        return {"error": f"Stage 2 (I2V) failed: {i2v_exec['error']}", "code": "PIPELINE_ERROR", "stage": "i2v"}

    timeout = args.timeout or 600
    i2v_output = execution.wait_for_completion(i2v_exec["prompt_id"], timeout_seconds=timeout, workflow=i2v_wf)

    if getattr(args, "output", None) and i2v_output.get("outputs"):
        asset_id = i2v_output["outputs"][0].get("asset_id")
        if asset_id:
            dl = execution.download_output(asset_id, args.output)
            i2v_output["downloaded"] = dl

    i2v_output["pipeline"] = "bio-to-video"
    i2v_output["stages"] = {
        "t2i": {"asset_id": creature_asset_id},
        "i2v": {"prompt_id": i2v_exec["prompt_id"]},
    }
    return i2v_output
