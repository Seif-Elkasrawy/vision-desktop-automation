# /// script
# dependencies = [
#   "pyautogui",
#   "opencv-python",
#   "pillow",
#   "google-genai",
#   "openai",
#   "requests",
#   "python-dotenv",
#   "numpy",
#   "pygetwindow"
# ]
# ///

import os
import time
import pyautogui
import PIL.ImageDraw as ImageDraw
 
from config import Config
from vision import VisionManager
from window_manager import WindowManager
from actions import AutomationActions
from data_client import DataClient

# Radius of the circle drawn on the annotated screenshot
ANNOTATION_RADIUS = 30

class NotepadOrchestrator:
    def __init__(self):
        self.config = Config()
        self.vision = VisionManager(self.config)
        self.window = WindowManager()
        self.actions = AutomationActions()
        self.data_client = DataClient()

    def _save_annotated_screenshot(self, screenshot, coords, post_id):
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
        draw.text((x + r + 4, y - 10), f"Notepad\n({x},{y})", fill="red")

        path = os.path.join(self.config.TARGET_DIR, f"screenshot_post_{post_id}.png")
        annotated.save(path)
        print(f"[Screenshot] Saved annotated screenshot -> {path}")
        return path

    def run(self):
        posts = self.data_client.fetch_posts()
 
        for post in posts:
            print(f"\nProcessing Post {post['id']}...")
 
            # Take the screenshot here so we own it for both grounding and saving
            screenshot = pyautogui.screenshot()
            coords = self.vision.get_coords_from_screenshot(screenshot)
 
            if not coords:
                print(f"Could not find Notepad icon for post {post['id']}")
                continue
 
            # Save the annotated screenshot to tjm-project
            self._save_annotated_screenshot(screenshot, coords, post['id'])
 
            # Launch Notepad, type and save the post
            post_path = os.path.join(
                self.config.TARGET_DIR, f"post_{post['id']}.txt"
            )
            self.actions.execute_flow(
                coords,
                f"Title: {post['title']}\n\n{post['body']}",
                post_path,
            )
 
            if self.window.validate_launch():
                print(f"Post {post['id']} saved -> {post_path}")
            else:
                print(f"Post {post['id']}: Notepad did not launch in time.")


if __name__ == "__main__":
    app = NotepadOrchestrator()
    app.run()