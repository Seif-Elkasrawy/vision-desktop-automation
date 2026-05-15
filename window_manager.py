import time
import pygetwindow as gw

class WindowManager:
    @staticmethod
    def validate_launch(title_keyword="Notepad", timeout=5):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if [w for w in gw.getAllTitles() if title_keyword in w]:
                return True
            time.sleep(0.5)
        return False