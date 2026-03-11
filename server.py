"""
Pasture Media MCP Server
========================
FastMCP server providing image generation, image editing, and video generation
tools powered by fal.ai. Includes automatic prompt refinement via Anthropic Claude.

Tools:
  - generate_image: Text-to-image via fal.ai Nano Banana Pro
  - edit_image: Image editing via fal.ai Nano Banana Pro Edit (with vision-based prompt refinement)
  - generate_video: Text-to-video via fal.ai Sora 2 Pro
"""

import json
import os

import anthropic
import fal_client
from fastmcp import FastMCP

mcp = FastMCP("Pasture Media")

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
# Tools
# ---------------------------------------------------------------------------


@mcp.tool
def generate_image(prompt: str, seed: int | None = None) -> str:
    """
    Generate a high-quality 4K image using fal.ai Nano Banana Pro.
    Use this when the user asks you to create, generate, or draw an image.

    Just pass the user's request as the prompt — prompt refinement is handled
    automatically by a specialized middleware. Do NOT try to craft a detailed
    image generation prompt yourself.

    Args:
        prompt: The user's image request as-is (e.g. 'a sad grape in a forest').
        seed: Optional seed for reproducible generation.

    Returns:
        JSON with the public fal.media image URL, model info, aspect ratio,
        resolution, and the refined prompt that was actually sent to the model.
    """
    refined = _refine_gen_prompt(prompt)

    input_args: dict = {
        "prompt": refined,
        "aspect_ratio": "auto",
        "resolution": "4K",
        "safety_tolerance": "6",
        "num_images": 1,
        "output_format": "png",
    }
    if seed is not None:
        input_args["seed"] = seed

    result = fal_client.subscribe("fal-ai/nano-banana-pro", arguments=input_args)

    images = result.get("images", [])
    if not images:
        return json.dumps({"error": "No image generated"})

    image_url = images[0].get("url", "")
    return json.dumps(
        {
            "success": True,
            "url": image_url,
            "seed": result.get("seed"),
            "model": "fal-ai/nano-banana-pro",
            "prompt": prompt,
            "refined_prompt": refined,
            "aspect_ratio": "auto",
            "resolution": "4K",
        }
    )


@mcp.tool
def edit_image(prompt: str, image_url: str) -> str:
    """
    Edit an existing image using fal.ai Nano Banana Pro Edit at 4K resolution.
    Use this when the user asks you to modify, edit, change, transform, or alter
    an existing image. A vision model automatically analyzes the source image and
    crafts an optimal editing prompt.

    IMPORTANT: image_url MUST be a publicly accessible URL — use the fal.media URL
    returned by generate_image or a previous edit_image call.

    Args:
        prompt: The user's edit request as-is (e.g. 'make the sky a sunset').
            Prompt refinement is automatic.
        image_url: Publicly accessible URL of the source image to edit.
            Use the fal.media URL from a previous generate_image or edit_image result.

    Returns:
        JSON with the edited image URL, model info, refined prompt, aspect ratio,
        resolution, and source image URL.
    """
    refined = _refine_edit_prompt(prompt, image_url)

    result = fal_client.subscribe(
        "fal-ai/nano-banana-pro/edit",
        arguments={
            "prompt": refined,
            "image_urls": [image_url],
            "aspect_ratio": "auto",
            "resolution": "4K",
            "safety_tolerance": "6",
            "num_images": 1,
            "output_format": "png",
        },
    )

    images = result.get("images", [])
    if not images:
        return json.dumps({"error": "No edited image returned"})

    edited_url = images[0].get("url", "")
    return json.dumps(
        {
            "success": True,
            "url": edited_url,
            "model": "fal-ai/nano-banana-pro/edit",
            "prompt": prompt,
            "refined_prompt": refined,
            "aspect_ratio": "auto",
            "resolution": "4K",
            "source_image": image_url,
        }
    )


@mcp.tool
def generate_video(prompt: str, aspect_ratio: str = "16:9") -> str:
    """
    Generate a video using Sora 2 Pro (OpenAI's latest video model via fal.ai).
    Use this when the user asks you to create, generate, or make a video.
    Video generation takes 1-3 minutes.

    Args:
        prompt: Detailed cinematic description of the video. Include camera angles,
            lighting, mood, motion, and scene details for best results.
        aspect_ratio: '16:9' for widescreen or '9:16' for vertical/mobile.
            Defaults to '16:9'.

    Returns:
        JSON with the video URL, model info, and prompt used.
    """
    valid = ["16:9", "9:16"]
    ar = aspect_ratio if aspect_ratio in valid else "16:9"

    result = fal_client.subscribe(
        "fal-ai/sora-2/text-to-video/pro",
        arguments={
            "prompt": prompt,
            "aspect_ratio": ar,
            "delete_video": False,
        },
    )

    video = result.get("video", {})
    if not video.get("url"):
        return json.dumps({"error": "No video generated"})

    return json.dumps(
        {
            "success": True,
            "url": video["url"],
            "model": "fal-ai/sora-2/text-to-video/pro",
            "prompt": prompt,
            "aspect_ratio": ar,
            "duration_seconds": video.get("duration", 4),
        }
    )


if __name__ == "__main__":
    mcp.run()
