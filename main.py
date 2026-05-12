# /// script
# dependencies = [
#   "pyautogui",
#   "opencv-python",
#   "pillow",
#   "openai",
#   "requests",
#   "python-dotenv",
#   "numpy"
# ]
# ///

import os
import time
import requests
import pyautogui
import base64
import json
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

# --- Config ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
TARGET_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "tjm-project")
os.makedirs(TARGET_DIR, exist_ok=True)


def encode_pil_to_base64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# --- Local Fallback (Section 5: Graceful Degradation) ---
def ground_locally_opencv():
    print("⚠️ API Unavailable. Running Local Grounding (OpenCV)...")
    try:
        # Take a fresh screenshot
        screen = np.array(pyautogui.screenshot())
        screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)

        # Load your icon cutout (MUST exist in folder)
        template = cv2.imread('notepad_icon.png', 0)
        if template is None:
            print("❌ Error: 'notepad_icon.png' missing! Please add the icon cutout to folder.")
            return None

        res = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > 0.7:  # 70% match threshold
            h, w = template.shape
            cx, cy = max_loc[0] + w // 2, max_loc[1] + h // 2

            # Save annotated screenshot (Deliverables Requirement #7)
            annotated = screen_bgr.copy()
            cv2.rectangle(annotated, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(TARGET_DIR, f"detection_{int(time.time())}.png"), annotated)

            print(f"✅ Icon Grounded at ({cx}, {cy})")
            return cx, cy
    except Exception as e:
        print(f"Local grounding failure: {e}")
    return None


# --- Paper-Based Grounding (ScreenSeekeR Approach) ---
def ground_notepad_icon_seeker():
    print(f"\n[{time.strftime('%H:%M:%S')}] Grounding Task Initiated...")

    screenshot = pyautogui.screenshot()

    try:
        # 1. Attempt API (Primary Strategy)
        base64_str = encode_pil_to_base64(screenshot)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": "Locate the Notepad desktop icon center coordinates (x,y) for a 1920x1080 screen. Output JSON: {'x': val, 'y': val}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_str}"}}
                ]
            }],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        return data['x'], data['y']

    except Exception as e:
        # Fallback to local if API has Quota Error, Network Error, or Timeout
        print(f"💡 Info: API Strategy skipped ({str(e)[:50]}...)")
        return ground_locally_opencv()


# --- Automation Flow ---
def start_automation():
    # Fetch Data Source (JSONPlaceholder)
    print("Connecting to Data Source...")
    posts = requests.get("https://jsonplaceholder.typicode.com/posts").json()[:10]

    for post in posts:
        # GROUND (Take screenshot -> identify icon)
        coords = ground_notepad_icon_seeker()

        if coords:
            # INTERACT (Double click to launch)
            pyautogui.doubleClick(coords)
            time.sleep(2)  # Wait for window to open

            # WORK (Type content)
            content = f"Title: {post['title']}\n\n{post['body']}"
            pyautogui.write(content, interval=0.01)

            # SAVE (Handle filesystem and keyboard shortcuts)
            pyautogui.hotkey('ctrl', 's')
            time.sleep(1)
            filepath = os.path.join(TARGET_DIR, f"post_{post['id']}.txt")
            if os.path.exists(filepath): os.remove(filepath)  # Handle existing files

            pyautogui.write(filepath)
            pyautogui.press('enter')
            time.sleep(1)

            # REPEAT (Close and fresh start for next grounding)
            pyautogui.hotkey('alt', 'f4')
            print(f"Successfully processed post_{post['id']}.txt")
            time.sleep(1)
        else:
            print("Skipping iteration: Notepad not visible.")


if __name__ == "__main__":
    start_automation()