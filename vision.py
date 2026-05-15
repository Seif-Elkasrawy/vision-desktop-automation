#Summary on how the vision manager works(also included in the README file):
# ------------------------------------------------------------------ #
# VisionManager uses a prompt chaining architecture to locate a target
# icon on screen. Rather than asking one prompt to do everything, the
# task is broken into three focused stages where each stage's output
# feeds into the next, making each step more accurate in isolation.
#
# Stage A — Planner:
#   Takes the full screenshot and asks Gemini to scan all quadrants of
#   the desktop and return up to 3 ranked candidate areas where the
#   target icon might be, along with a neighboring icon for context.
#   Output: a list of candidate bounding boxes with probabilities.
#
# Stage B — Grounder:
#   Takes each candidate crop from the Planner and asks Gemini to
#   locate the icon precisely within that smaller image. Only works on
#   a zoomed-in region so it doesn't have to reason about the full
#   screen. Output: a tight bounding box and confidence score.
#
# Stage C — Verifier:
#   Takes the predicted pixel coordinates from the Grounder, draws a
#   red circle at that point, and asks Gemini to confirm whether the
#   circle is centred on the correct icon. If it is off-centre, the
#   Verifier returns a corrected box. Output: verified or corrected
#   final coordinates.
#
# Fallback chain (in order if any stage fails):
#   1. Gemini (Planner → Grounder → Verifier) — primary pipeline.
#      Each individual Gemini API call is retried up to 4 times with
#      exponential backoff if a transient error occurs (503, 429 etc),
#      doubling the wait time between each attempt.
#   2. GPT-4o — single prompt asking OpenAI to locate the icon directly
#      from the full screenshot, used if Gemini fails entirely.
#   3. OpenCV template matching — purely local, no AI, matches a saved
#      icon image against the screenshot using pixel similarity.
#
# The full fallback chain is retried up to 3 times before giving up,
# with a fresh screenshot taken on each retry.
#
# Coordinate caching:
#   Once a target is successfully grounded, its coordinates are cached
#   by the orchestrator and reused for subsequent posts, skipping the
#   entire pipeline. If a click using cached coordinates fails to open
#   the target (validated by WindowManager), the cache is invalidated
#   and the full pipeline runs again to find the updated position.
# ------------------------------------------------------------------ #

import os
import time
import json
import base64
import cv2
import numpy as np
import pyautogui
from io import BytesIO
from openai import OpenAI
from google import genai
from google.genai import types # Added for JSON mode

from prompts import PLANNER, GROUNDER, VERIFIER
from constants import RETRY_DELAY_S, VERIFY_RADIUS

class VisionManager:
    def __init__(self, config):
        self.config = config
        self.gemini_client = genai.Client(api_key=config.GEMINI_KEY)
        self.openai_client = OpenAI(api_key=config.OPENAI_KEY)
        self.planner_model = "gemini-2.5-flash" 

    # ------------------------------------------------------------------ #
    # Primary: Gemini ScreenSeekeR (planner -> multi-candidate -> grounder)
    # ------------------------------------------------------------------ #
    def ground_ai_gemini(self, screenshot, target: str = "Notepad"):
        w, h = screenshot.size
 
        # ---- Stage A: Position Inference with neighbor context ---------- #
        # Key improvements:
        #  * Returns up to 3 ranked candidate areas instead of one
        #  * Neighbor inference: model names adjacent icons to anchor reasoning
        #  * Explicit instruction to look at the image, not guess from priors
        try:
            # Stage A – Planner
            planner_text = self._gemini_call_with_retry(
                prompt=PLANNER["prompt"].format(target=target),
                image=screenshot,
                schema=PLANNER["schema"],
            )
            plan       = json.loads(planner_text)
            reasoning  = plan.get("reasoning", "")
            candidates = plan.get("candidates", [])
 
            print(f"[Planner] Reasoning: {reasoning}")
            print(f"[Planner] {len(candidates)} candidate(s).")
 
            if not candidates:
                print("[Planner] No candidates returned.")
                return None
 
            # ---- Stage B: Grounder on each candidate; keep best verified result -- #
            best_coords     = None
            best_confidence = 0.0
 
            for idx, cand in enumerate(candidates):
                box  = cand.get("area_box", [])
                prob = cand.get("probability", 0.0)
                nbr  = cand.get("neighbor", "unknown")
 
                if len(box) != 4:
                    print(f"[Planner] Candidate {idx} has malformed box {box}, skipping.")
                    continue
 
                xmin_n, ymin_n, xmax_n, ymax_n = [max(0, min(1000, v)) for v in box]
                left   = int(xmin_n * w / 1000)
                top    = int(ymin_n * h / 1000)
                right  = int(xmax_n * w / 1000)
                bottom = int(ymax_n * h / 1000)
 
                if right - left < 20 or bottom - top < 20:
                    print(f"[Planner] Candidate {idx} crop too small, skipping.")
                    continue
 
                print(f"[Planner] Candidate {idx}: prob={cand.get('probability',0):.2f}, "
                      f"neighbor='{cand.get('neighbor','?')}', "
                      f"px=({left},{top})->({right},{bottom})")
 
                crop = screenshot.crop((left, top, right, bottom))
 
                # Grounder: where in the crop is the icon?
                raw_coords, confidence = self._ground_in_crop(crop, target, idx)
                if raw_coords is None:
                    continue
 
                cx_crop, cy_crop = raw_coords
                screen_x = left + cx_crop
                screen_y = top  + cy_crop
 
                # ---- Stage C: Verification ------------- #
                # Crop a small window around the predicted point and ask
                # Gemini: "is this actually the target icon?"
                verified, new_x, new_y = self._verify_prediction(
                    screenshot, screen_x, screen_y, target, w, h
                )
 
                if not verified:
                    print(f"[Verify] Candidate {idx} failed verification — skipping.")
                    continue
 
                # Use refined coordinates if the verifier found a better centre
                if new_x is not None:
                    screen_x, screen_y = new_x, new_y
 
                if confidence > best_confidence:
                    best_coords     = (screen_x, screen_y)
                    best_confidence = confidence
                    print(f"[Grounder] Verified best: confidence={confidence:.2f}, "
                          f"screen=({screen_x},{screen_y})")
 
            if best_coords is None:
                print("[Grounder] No candidate passed verification.")
                return None
 
            print(f"[Gemini] Final '{target}' at {best_coords} "
                  f"(confidence={best_confidence:.2f})")
            return best_coords
 
        except Exception as e:
            print(f"[Gemini ScreenSeekeR Error] {e}")
            return None
 
    def _ground_in_crop(self, crop, target: str, candidate_idx: int):
        """
        Run the grounder on a single crop image.
        Returns (center_xy_in_crop_pixels, confidence) or (None, 0.0).
        """
        crop_w, crop_h = crop.size
 
        try:
            resp = json.loads(self._gemini_call_with_retry(
                prompt=GROUNDER["prompt"].format(target=target),
                image=crop,
                schema=GROUNDER["schema"],
            ))
 
            confidence = float(resp.get("confidence", 0.0))
            found      = resp.get("found", False)
            label_seen = resp.get("label_seen", "")
 
            print(f"[Grounder] Candidate {candidate_idx}: found={found}, "
                  f"label='{label_seen}', confidence={confidence:.2f}")
 
            if not found or confidence < 0.5:
                return None, 0.0
 
            box = resp.get("target_box", [])
            if len(box) != 4:
                return None, 0.0
 
            t_xmin, t_ymin, t_xmax, t_ymax = [max(0, min(1000, v)) for v in box]
            cx = int(((t_xmin + t_xmax) / 2) * crop_w / 1000)
            cy = int(((t_ymin + t_ymax) / 2) * crop_h / 1000)
            return (cx, cy), confidence
 
        except Exception as e:
            print(f"[Grounder] Error on candidate {candidate_idx}: {e}")
            return None, 0.0
        
        # ------------------------------------------------------------------ #
    # Stage C: Verification (implements the paper's Result Checking prompt)
    # ------------------------------------------------------------------ #
    def _verify_prediction(self, screenshot, pred_x, pred_y, target, img_w, img_h):
        """
        Crop a small window around (pred_x, pred_y) and ask Gemini whether
        the element inside the red-box area is really the target icon.
 
        Returns (verified: bool, refined_x or None, refined_y or None).
        """
        r = VERIFY_RADIUS # pixels around the predicted centre to include in the verification crop
        vx1 = max(0,     pred_x - r)
        vy1 = max(0,     pred_y - r)
        vx2 = min(img_w, pred_x + r)
        vy2 = min(img_h, pred_y + r)
 
        verify_crop = screenshot.crop((vx1, vy1, vx2, vy2))
        vcw, vch    = verify_crop.size
 
        # Draw a small red circle at the predicted centre so the model can
        # see exactly which pixel we are claiming is the icon
        import PIL.ImageDraw as ImageDraw
        annotated = verify_crop.copy()
        draw = ImageDraw.Draw(annotated)
        cx_in_crop = pred_x - vx1
        cy_in_crop = pred_y - vy1
        draw.ellipse(
            [cx_in_crop - 8, cy_in_crop - 8, cx_in_crop + 8, cy_in_crop + 8],
            outline="red", width=3
        )
 
        try:
            resp = json.loads(self._gemini_call_with_retry(
                prompt=VERIFIER["prompt"].format(target=target),
                image=annotated,
                schema=VERIFIER["schema"],
            ))
 
            result_str  = resp.get("result", "target_not_found")
            description = resp.get("description", "")
            print(f"[Verify] result='{result_str}' | {description}")
 
            if result_str == "is_target":
                return True, None, None  # prediction is correct as-is
 
            if result_str == "target_elsewhere":
                cbox = resp.get("corrected_box")
                if cbox and len(cbox) == 4:
                    cx_n = (cbox[0] + cbox[2]) / 2
                    cy_n = (cbox[1] + cbox[3]) / 2
                    refined_x = vx1 + int(cx_n * vcw / 1000)
                    refined_y = vy1 + int(cy_n * vch / 1000)
                    print(f"[Verify] Corrected to ({refined_x}, {refined_y})")
                    return True, refined_x, refined_y
 
            # "target_not_found" — this candidate is wrong
            return False, None, None
 
        except Exception as e:
            print(f"[Verify] Error: {e} — accepting grounder result without verification.")
            # On error, accept the grounder result to avoid blocking the pipeline
            return True, None, None
 
    # ------------------------------------------------------------------ #
    # Helper: Gemini call with exponential-backoff retry
    # ------------------------------------------------------------------ #
    def _gemini_call_with_retry(self, prompt, image, schema, max_retries=4):
        delay = 1.0
        last_exc = None
        for attempt in range(max_retries):
            try:
                resp = self.gemini_client.models.generate_content(
                    model=self.planner_model,
                    contents=[prompt, image],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )
                return resp.text
            except Exception as e:
                last_exc = e
                err_str = str(e)
                if any(c in err_str for c in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")):
                    print(f"[Gemini] Transient error (attempt {attempt+1}/{max_retries}): {e}. "
                          f"Retrying in {delay:.0f}s...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise
        raise last_exc
 
    # ---------------------------------------------------------------------- #
    # OpenCV template-matching fallback
    # ---------------------------------------------------------------------- #
    def ground_locally_opencv(self):
        print("⚠️  Running Local Grounding (OpenCV template match)…")
        try:
            screen = np.array(pyautogui.screenshot())
            screen_bgr  = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
            screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
            template = cv2.imread(self.config.ICON_PATH, 0)
 
            if template is None:
                print(f"[OpenCV] Template not found at: {self.config.ICON_PATH}")
                return None
 
            # Only search the desktop area (exclude taskbar at bottom ~10%)
            res = cv2.matchTemplate(screen_gray[0:int(screen_gray.shape[0]*1), :], template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > 0.8:
                h, w = template.shape
                return max_loc[0] + w // 2, max_loc[1] + h // 2
        except Exception as e:
            print(f"Local error: {e}")
        return None
    
    # ---------------------------------------------------------------------- #
    # GPT-4o fallback
    # ---------------------------------------------------------------------- #
    def _encode_pil_to_base64(self, pil_img):
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
 
    def ground_ai_openai(self, screenshot, target: str = "Notepad"):
        print("⚠️  Running OpenAI GPT-4o Grounding…")
        try:
            b64 = self._encode_pil_to_base64(screenshot)
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f'Locate the "{target}" desktop icon on this Windows screenshot. '
                                f'Return ONLY a JSON object with integer keys "x" and "y" '
                                f'representing the pixel centre of the icon logo (not the text label). '
                                f'Example: {{"x": 120, "y": 95}}'
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            x, y = int(data["x"]), int(data["y"])
            print(f"[OpenAI] Grounded '{target}' at: ({x}, {y})")
            return x, y
        except Exception as e:
            print(f"[OpenAI Error] {e}")
            return None

    # ------------------------------------------------------------------ #
    # Public entry points
    # ------------------------------------------------------------------ #
    def get_coords_from_screenshot(self, screenshot, target: str = "Notepad"):
        """
        Ground `target` using an already-captured screenshot.
        The orchestrator calls this so it can also save the same screenshot.
        Retries up to 3 times; on each retry a fresh screenshot is taken.
        """
        # First attempt uses the caller-supplied screenshot
        for attempt in range(1, 4):
            print(f"\n--- Grounding attempt {attempt}/3 for '{target}' ---")
            img = screenshot if attempt == 1 else pyautogui.screenshot()
            coords = (
                self.ground_ai_gemini(img, target)
                or self.ground_ai_openai(img, target)
                or self.ground_locally_opencv()
            )
            if coords:
                return coords
            time.sleep(RETRY_DELAY_S)
 
        print(f"[VisionManager] All grounding attempts failed for '{target}'.")
        return None
 
    def get_coords(self, target: str = "Notepad"):
        """
        Convenience wrapper that takes its own screenshot.
        Useful for standalone use or testing.
        """
        return self.get_coords_from_screenshot(pyautogui.screenshot(), target)