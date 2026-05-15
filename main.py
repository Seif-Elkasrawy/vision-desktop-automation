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
 
from config import Config
from vision import VisionManager
from window_manager import WindowManager
from actions import AutomationActions
from data_client import DataClient
from utils import _save_annotated_screenshot

class NotepadOrchestrator:
    def __init__(self):
        self.config = Config()
        self.vision = VisionManager(self.config)
        self.window = WindowManager()
        self.actions = AutomationActions()
        self.data_client = DataClient()
        self.cached_coords = None  # populated after first successful grounding

    def run(self):
        target = "Notepad"
        posts = self.data_client.fetch_posts()

        for post in posts:
            print(f"\nProcessing Post {post['id']}...")

            screenshot = pyautogui.screenshot()

            if self.cached_coords:
                print(f"[Cache] Using cached coords {self.cached_coords}, skipping Gemini.")
                coords = self.cached_coords
            else:
                coords = self.vision.get_coords_from_screenshot(screenshot, target)
                if coords:
                    self.cached_coords = coords
                    print(f"[Cache] Coords cached at {coords}.")

            if not coords:
                print(f"Could not find {target} icon for post {post['id']}")
                continue

            _save_annotated_screenshot(screenshot, coords, post['id'], self.config.TARGET_DIR, target)

            post_path = os.path.join(self.config.TARGET_DIR, f"post_{post['id']}.txt")
            self.actions.execute_flow(
                coords,
                f"Title: {post['title']}\n\n{post['body']}",
                post_path,
            )

            if self.window.validate_launch(target):
                print(f"Post {post['id']} saved -> {post_path}")
            else:
                print(f"Post {post['id']}: {target} did not launch in time.")

                if self.cached_coords:
                    print(f"[Cache] Launch failed with cached coords — invalidating cache and retrying.")
                    self.cached_coords = None
                    screenshot = pyautogui.screenshot()
                    coords = self.vision.get_coords_from_screenshot(screenshot, target)

                    if not coords:
                        print(f"[Retry] Could not re-ground {target} for post {post['id']}, skipping.")
                        continue

                    self.cached_coords = coords
                    print(f"[Cache] New coords cached at {coords}.")
                    self.actions.execute_flow(
                        coords,
                        f"Title: {post['title']}\n\n{post['body']}",
                        post_path,
                    )

                    if self.window.validate_launch(target):
                        print(f"Post {post['id']} saved -> {post_path}")
                    else:
                        print(f"Post {post['id']}: failed after re-grounding, skipping.")


if __name__ == "__main__":
    app = NotepadOrchestrator()
    app.run()