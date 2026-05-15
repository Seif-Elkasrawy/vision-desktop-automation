import os
from PIL import ImageDraw
from constants import ANNOTATION_RADIUS


def _save_annotated_screenshot(screenshot, coords, post_id, target_dir, target):
    """
    Draw a red circle + crosshair at the grounded coordinates and save
    the annotated image to the tjm-project folder as
    screenshot_post_{post_id}.png
    """
    annotated = screenshot.copy()
    draw = ImageDraw.Draw(annotated)
    x, y = coords
    r = ANNOTATION_RADIUS

    # Outer circle
    draw.ellipse([x - r, y - r, x + r, y + r], outline="red", width=3)
    # Crosshair
    draw.line([x - r, y, x + r, y], fill="red", width=2)
    draw.line([x, y - r, x, y + r], fill="red", width=2)
    # Label
    draw.text((x + r + 4, y - 10), f"{target}\n({x},{y})", fill="red")

    path = os.path.join(target_dir, f"screenshot_post_{post_id}.png")
    annotated.save(path)
    print(f"[Screenshot] Saved annotated screenshot -> {path}")
    return path