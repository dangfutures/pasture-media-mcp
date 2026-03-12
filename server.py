"""
Pasture Media MCP Server
========================
FastMCP server providing image generation, image editing, and video generation
tools powered by fal.ai. Includes automatic prompt refinement via Anthropic Claude.

Tools:
  - generate_image: Text-to-image via fal.ai Nano Banana Pro (4K)
  - edit_image: Image editing via fal.ai Nano Banana Pro Edit (4K, vision-based prompt refinement)
  - generate_video: Text-to-video via fal.ai Sora 2 Pro (1080p, up to 12s)
"""

import json
import os
from typing import Literal

import anthropic
import fal_client
from fastmcp import FastMCP

mcp = FastMCP(
    "Pasture Media",
    instructions="""
    Media generation server for creating images, editing images, and generating videos.

    Tool selection guide:
    - generate_image: Use when asked to create, generate, draw, or make a NEW image from scratch.
    - edit_image: Use when asked to modify, edit, change, or transform an EXISTING image.
      If the user wants to edit the last generated image, omit image_url — it auto-chains.
    - generate_video: Use when asked to create, generate, or make a video clip.
      Video generation takes 1-3 minutes. Warn the user about wait time.

    All image tools produce 4K output. Prompt refinement is automatic — pass the
    user's request as-is without rewriting it into a detailed prompt yourself.
    """,
)

# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

ImageAspectRatio = Literal[
    "auto", "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16"
]

VideoAspectRatio = Literal["16:9", "9:16"]

ImageResolution = Literal["1K", "2K", "4K"]

VideoDuration = Literal[4, 8, 12]

# ---------------------------------------------------------------------------
# Prompt refinement system prompts
# ---------------------------------------------------------------------------

IMAGE_GEN_SYSTEM_PROMPT = """You are an expert image generation prompt engineer for the Nano Banana Pro model (fal.ai).

Your job: given a user's image request, craft ONE optimal generation prompt.

## Prompt Rules
1. Use the formula: [Subject] + [Action] + [Location/Context] + [Composition] + [Style]
2. Be specific — provide concrete details on subject, lighting, and composition.
3. Use positive framing: say "empty street" not "no cars."
4. Lead with a strong descriptive statement, not a verb command.
5. Control the camera: use photographic and cinematic terms like "low angle," "aerial view," "close-up."
6. Specify style: "cartoon style," "photorealistic," "oil painting," "fashion editorial," etc.

## Creative Director Controls (use when relevant)
- Lighting: "three-point softbox", "golden hour backlighting", "chiaroscuro high contrast"
- Camera/Lens: "shallow depth of field (f/1.8)", "wide-angle lens", "macro lens"
- Color grading: "muted teal tones", "1980s color film grain", "high saturation editorial"
- Materiality: Be precise — "navy blue tweed" not just "jacket"

## Text Rendering (if text is requested)
- Enclose text in quotes: "Happy Birthday"
- Specify the font: "bold, white, sans-serif font"

## Output
Respond with ONLY the generation prompt — no explanation, no markdown, no quotes. Just the raw prompt text."""

IMAGE_EDIT_SYSTEM_PROMPT = """You are an expert image editing prompt engineer for the Nano Banana Pro model (fal.ai).

Your job: given a source image and a user's edit request, craft ONE optimal editing prompt for fal.ai's image editing API.

## Prompt Rules
1. Describe what changes and what stays the same — be explicit about preserving elements.
2. Use positive framing: say "empty street" not "no cars."
3. Lead with a strong verb: edit, transform, change, replace, add, remove.
4. Use semantic masking — define text-based targets for specific regions (e.g. "the sky", "the person's shirt").
5. Be specific about lighting, color, materials, and style when relevant.
6. For style transfer: describe the target style with concrete references.
7. For adding elements: describe placement, scale, and how it integrates with existing content.
8. For removing elements: state what to remove and what should fill the space.

## Creative Director Controls (use when relevant)
- Lighting: "three-point softbox", "golden hour backlighting", "chiaroscuro high contrast"
- Camera/Lens: "shallow depth of field (f/1.8)", "wide-angle lens", "macro lens"
- Color grading: "muted teal tones", "1980s color film grain", "high saturation editorial"
- Materiality: Be precise — "navy blue tweed" not just "jacket"

## Output
Respond with ONLY the editing prompt — no explanation, no markdown, no quotes. Just the raw prompt text."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_anthropic() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


def _refine_gen_prompt(user_prompt: str) -> str:
    """Use Claude to refine a casual image request into an optimal fal.ai prompt."""
    try:
        client = _get_anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=IMAGE_GEN_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Image request: {user_prompt}"}],
        )
        text = response.content[0].text if response.content else ""
        if text.strip():
            return text.strip()
    except Exception as e:
        print(f"[generate_image] Prompt refinement failed: {e}")
    return user_prompt


def _refine_edit_prompt(user_prompt: str, image_url: str) -> str:
    """Use Claude vision to see the image and craft an optimal edit prompt."""
    try:
        client = _get_anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=IMAGE_EDIT_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "url", "url": image_url},
                        },
                        {
                            "type": "text",
                            "text": f"Edit request: {user_prompt}",
                        },
                    ],
                }
            ],
        )
        text = response.content[0].text if response.content else ""
        if text.strip():
            return text.strip()
    except Exception as e:
        print(f"[edit_image] Vision prompt refinement failed: {e}")
    return user_prompt


# ---------------------------------------------------------------------------
# State: track last generated/edited image URL for seamless editing
# ---------------------------------------------------------------------------

_last_image_url: str | None = None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def generate_image(
    prompt: str,
    aspect_ratio: ImageAspectRatio = "auto",
    resolution: ImageResolution = "4K",
    seed: int | None = None,
) -> str:
    """
    Generate a high-quality image from a text description.

    Use this when the user asks you to create, generate, or draw an image.
    Prompt refinement is handled automatically — pass the user's request as-is.

    Args:
        prompt: The user's image request as-is (e.g. "a sad grape in a forest").
            Do NOT try to craft a detailed image generation prompt yourself.
        aspect_ratio: Shape of the output image. Use "auto" to let the model
            decide, or pick a specific ratio for the user's needs:
            - Landscape: "21:9" (ultrawide), "16:9" (widescreen), "3:2", "4:3"
            - Square: "1:1"
            - Portrait: "4:5" (social), "3:4", "2:3", "9:16" (vertical/mobile)
            Defaults to "auto".
        resolution: Image resolution. "4K" for maximum quality, "2K" for
            faster generation, "1K" for drafts/thumbnails. Defaults to "4K".
        seed: Optional seed for reproducible generation.

    Returns:
        JSON with the public image URL, model info, aspect ratio, resolution,
        and the refined prompt that was actually sent to the model.

    Example:
        generate_image("a goat in a courtroom wearing a tiny lawyer wig", aspect_ratio="16:9")
        → {"success": true, "url": "https://fal.media/files/...", "aspect_ratio": "16:9", ...}
    """
    refined = _refine_gen_prompt(prompt)

    input_args: dict = {
        "prompt": refined,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "safety_tolerance": "6",
        "num_images": 1,
        "output_format": "png",
    }
    if seed is not None:
        input_args["seed"] = seed

    result = fal_client.subscribe("fal-ai/nano-banana-pro", arguments=input_args)

    images = result.get("images", [])
    if not images:
        return json.dumps(
            {
                "error": "No image generated. The model returned an empty result. "
                "Try simplifying the prompt or using a different aspect_ratio."
            }
        )

    image_url = images[0].get("url", "")

    # Track for seamless edit_image follow-ups
    global _last_image_url
    _last_image_url = image_url

    return json.dumps(
        {
            "success": True,
            "url": image_url,
            "width": images[0].get("width"),
            "height": images[0].get("height"),
            "seed": result.get("seed"),
            "model": "fal-ai/nano-banana-pro",
            "prompt": prompt,
            "refined_prompt": refined,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
        }
    )


@mcp.tool()
def edit_image(
    prompt: str,
    image_url: str | None = None,
    aspect_ratio: ImageAspectRatio = "auto",
    resolution: ImageResolution = "4K",
) -> str:
    """
    Edit an existing image using vision-guided AI.

    Use this when the user asks you to modify, edit, change, transform, or alter
    an existing image. A vision model automatically analyzes the source image and
    crafts an optimal editing prompt — just pass the user's request as-is.

    If image_url is omitted, the last generated or edited image is used automatically
    (seamless chaining: generate -> edit -> edit -> ...).

    Args:
        prompt: The user's edit request as-is (e.g. "make the sky a sunset").
            Do NOT try to craft a detailed editing prompt yourself.
        image_url: Publicly accessible URL of the source image to edit.
            If omitted, uses the last generated/edited image automatically.
        aspect_ratio: Shape of the output image. Use "auto" to preserve the
            source image's aspect ratio, or pick a specific ratio to crop/reshape.
            Defaults to "auto".
        resolution: Image resolution. "4K" for maximum quality, "2K" for
            faster generation, "1K" for drafts/thumbnails. Defaults to "4K".

    Returns:
        JSON with the edited image URL, model info, refined prompt, aspect ratio,
        resolution, and source image URL.

    Example:
        edit_image("make it nighttime with neon lights")
        → {"success": true, "url": "https://fal.media/files/...", "source_image": "...", ...}
    """
    global _last_image_url

    # Fall back to last generated/edited image if no URL provided
    if not image_url:
        if not _last_image_url:
            return json.dumps(
                {
                    "error": "No image_url provided and no previous image to edit. "
                    "Either provide an image_url parameter, or call generate_image first "
                    "to create an image that can then be edited."
                }
            )
        image_url = _last_image_url

    refined = _refine_edit_prompt(prompt, image_url)

    result = fal_client.subscribe(
        "fal-ai/nano-banana-pro/edit",
        arguments={
            "prompt": refined,
            "image_urls": [image_url],
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "safety_tolerance": "6",
            "num_images": 1,
            "output_format": "png",
        },
    )

    images = result.get("images", [])
    if not images:
        return json.dumps(
            {
                "error": "No edited image returned. The model may have had trouble "
                "with the edit request. Try rephrasing the prompt or using a "
                "different source image."
            }
        )

    edited_url = images[0].get("url", "")

    # Track for subsequent edits
    _last_image_url = edited_url

    return json.dumps(
        {
            "success": True,
            "url": edited_url,
            "width": images[0].get("width"),
            "height": images[0].get("height"),
            "model": "fal-ai/nano-banana-pro/edit",
            "prompt": prompt,
            "refined_prompt": refined,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "source_image": image_url,
        }
    )


@mcp.tool()
def generate_video(
    prompt: str,
    aspect_ratio: VideoAspectRatio = "16:9",
    duration: VideoDuration = 4,
) -> str:
    """
    Generate a video clip with audio using Sora 2 Pro (OpenAI's latest video model).

    Use this when the user asks you to create, generate, or make a video.
    Video generation takes 1-3 minutes — warn the user about wait time.

    Args:
        prompt: Detailed cinematic description of the video. Include camera angles,
            lighting, mood, motion, and scene details for best results. Unlike image
            generation, video prompts are NOT auto-refined — write a good one.
        aspect_ratio: "16:9" for widescreen or "9:16" for vertical/mobile.
            Defaults to "16:9".
        duration: Length of the video in seconds. Choose 4, 8, or 12.
            Longer videos cost more and take longer to generate. Defaults to 4.

    Returns:
        JSON with the video URL, model info, aspect ratio, and duration.

    Example:
        generate_video(
            "Aerial drone shot of a goat standing on a mountain peak at sunrise, "
            "golden light, clouds below, cinematic orchestral score",
            aspect_ratio="16:9",
            duration=8
        )
        → {"success": true, "url": "https://...", "duration_seconds": 8, ...}
    """
    result = fal_client.subscribe(
        "fal-ai/sora-2/text-to-video/pro",
        arguments={
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "resolution": "1080p",
            "delete_video": False,
        },
    )

    video = result.get("video", {})
    if not video.get("url"):
        return json.dumps(
            {
                "error": "No video generated. The model returned an empty result. "
                "Try simplifying the prompt or reducing the duration."
            }
        )

    return json.dumps(
        {
            "success": True,
            "url": video["url"],
            "model": "fal-ai/sora-2/text-to-video/pro",
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": "1080p",
            "duration_seconds": video.get("duration") or duration,
        }
    )


if __name__ == "__main__":
    mcp.run()
