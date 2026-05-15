import os
import time
import pyautogui

FOCUS_DELAY_S   = 1.0   # Wait for Notepad window to gain focus
SAVE_DIALOG_S   = 1.0   # Wait for Save As dialog to appear
CLOSE_DELAY_S   = 1.0   # Wait after Alt+F4 before next iteration
POLL_INTERVAL_S = 0.5   # WindowManager polling cadence

class AutomationActions:
    def execute_flow(self, coords, content, filepath):
        self._open_notepad(coords)
        self._type_content(content)
        self._save_file(filepath)
        self._close_window()

    def _open_notepad(self, coords):
        pyautogui.doubleClick(coords)
        time.sleep(FOCUS_DELAY_S)

    def _type_content(self, content):
        pyautogui.write(content, interval=0.01)

    def _save_file(self, filepath):
        pyautogui.hotkey('ctrl', 's')
        time.sleep(SAVE_DIALOG_S)
        if os.path.exists(filepath):
            os.remove(filepath)
        pyautogui.write(filepath)
        pyautogui.press('enter')

    def _close_window(self):
        time.sleep(CLOSE_DELAY_S)
        pyautogui.hotkey('alt', 'f4')