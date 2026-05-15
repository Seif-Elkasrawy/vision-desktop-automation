import time
import pygetwindow as gw
from constants import POLL_INTERVAL_S

class WindowManager:
    @staticmethod
    def validate_launch(title_keyword="Notepad", timeout=5):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if [w for w in gw.getAllTitles() if title_keyword in w]:
                return True
            time.sleep(POLL_INTERVAL_S)
        return False