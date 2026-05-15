import os
import time
import pyautogui

class AutomationActions:
    @staticmethod
    def execute_flow(coords, content, filepath):
        pyautogui.doubleClick(coords)
        time.sleep(1) # Wait for focus
        
        # Write content
        pyautogui.write(content, interval=0.01)
        
        # Save sequence
        pyautogui.hotkey('ctrl', 's')
        time.sleep(1)
        
        if os.path.exists(filepath):
            os.remove(filepath)

        pyautogui.write(filepath)
        pyautogui.press('enter')
        time.sleep(1)
        
        # Close
        pyautogui.hotkey('alt', 'f4')