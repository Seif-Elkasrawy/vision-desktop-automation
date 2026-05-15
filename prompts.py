"""
prompts.py : All Gemini prompt strings and their JSON schemas.

Each entry is a plain dict with keys:
  "prompt"  : the f-string template (call .format(target=...) before use)
  "schema"  : the response_schema dict passed to GenerateContentConfig
"""

# ------------------------------------------------------------------ #
# Planner: scan the full screenshot, return ranked candidate areas
# ------------------------------------------------------------------ #
PLANNER = {
    "prompt": """
You are a GUI grounding assistant examining a REAL screenshot of a Windows desktop.

Target: find the desktop shortcut icon whose label contains "{target}".

IMPORTANT: Do NOT assume a default position (e.g. bottom-left). You MUST reason
from what you actually see in the screenshot.

Step 1 - Orient yourself: identify the Taskbar, the Recycle Bin, and any other
         visible desktop icons. Note their approximate positions.
Step 2 - Scan ALL quadrants of the desktop area (above the taskbar) carefully.
         Describe what icons you see in each quadrant.
Step 3 - Propose up to 3 candidate areas where "{target}" might be, ranked by
         probability. For each, name a neighboring icon you can see nearby to
         confirm the region.

Return JSON with this EXACT structure:
{{
  "reasoning": "<step-by-step description of what you see and where you looked>",
  "candidates": [
    {{
      "area_box": [xmin, ymin, xmax, ymax],
      "neighbor": "<name of a visible nearby icon>",
      "probability": <float 0.0-1.0>
    }}
  ]
}}

Rules:
- area_box values are integers 0-1000 (normalised: 0=left/top edge, 1000=right/bottom edge).
- List candidates in descending probability order.
- Each box should be roughly 300-600 units wide/tall.
- Return an empty list if you truly cannot locate any plausible area.
""",
    "schema": {
        "type": "OBJECT",
        "properties": {
            "reasoning": {"type": "STRING"},
            "candidates": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "area_box":    {"type": "ARRAY", "items": {"type": "NUMBER"}},
                        "neighbor":    {"type": "STRING"},
                        "probability": {"type": "NUMBER"},
                    },
                    "required": ["area_box", "probability"],
                },
            },
        },
        "required": ["candidates"],
    },
}

# ------------------------------------------------------------------ #
# Grounder: locate the icon precisely inside a candidate crop
# ------------------------------------------------------------------ #
GROUNDER = {
    "prompt": """
You are a GUI grounding assistant.

Look carefully at this zoomed-in crop of a Windows desktop and locate the
"{target}" shortcut icon.

Step 1 - List every icon label and graphic you can see in this image.
Step 2 - Identify which one matches "{target}". If none match, set found=false
         and confidence=0.
Step 3 - Return the bounding box of the ICON GRAPHIC ONLY (not its text label).

Return JSON:
{{
  "found": true | false,
  "label_seen": "<exact text label visible under or near the matching icon>",
  "target_box": [xmin, ymin, xmax, ymax],
  "confidence": <float 0.0-1.0>
}}

Coordinates are integers 0-1000 normalised to the WIDTH and HEIGHT of this crop.
Set confidence=0 and found=false if "{target}" is not present in this image.
""",
    "schema": {
        "type": "OBJECT",
        "properties": {
            "found":      {"type": "BOOLEAN"},
            "label_seen": {"type": "STRING"},
            "target_box": {"type": "ARRAY", "items": {"type": "NUMBER"}},
            "confidence": {"type": "NUMBER"},
        },
        "required": ["found", "target_box", "confidence"],
    },
}

# ------------------------------------------------------------------ #
# Verifier: confirm the grounder's prediction is correct
# ------------------------------------------------------------------ #
VERIFIER = {
    "prompt": """
You are given a small crop of a Windows desktop.
A red circle marks the element we believe is the "{target}" icon.

Please evaluate whether the circled element is correct:

Step 1 - Describe the visible content: what icons, labels, or UI elements are present?
Step 2 - Determine which of the following applies to the circled element:
   - "is_target":        the circled element IS the "{target}" icon
   - "target_elsewhere": the circled element is NOT "{target}", but "{target}" IS
                          visible somewhere else in this crop
   - "target_not_found": "{target}" is not visible anywhere in this crop

Step 3 - If the result is "target_elsewhere", provide the corrected bounding box.

Step 4 - If the result is "is_target", verify the red circle is centred on the
         ICON GRAPHIC itself (not the text label beneath it, not an adjacent icon).
         If the circle is off-centre, set result to "target_elsewhere" and provide
         a corrected_box tightly around the icon graphic centre.

Return JSON:
{{
  "result": "is_target" | "target_elsewhere" | "target_not_found",
  "description": "<brief description of what you see>",
  "corrected_box": [xmin, ymin, xmax, ymax] | null
}}

corrected_box is in 0-1000 coordinates normalised to this crop's width and height.
Only populate it when result is "target_elsewhere".
""",
    "schema": {
        "type": "OBJECT",
        "properties": {
            "result":        {"type": "STRING"},
            "description":   {"type": "STRING"},
            "corrected_box": {"type": "ARRAY", "items": {"type": "NUMBER"}},
        },
        "required": ["result"],
    },
}