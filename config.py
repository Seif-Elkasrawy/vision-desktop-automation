import os
from dotenv import load_dotenv
from dataclasses import dataclass

@dataclass
class Config:
    def __init__(self):
        load_dotenv()
        self.TARGET_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "tjm-project")
        self.GEMINI_KEY = os.getenv("GEMINI_API_KEY")
        self.OPENAI_KEY = os.getenv("OPENAI_API_KEY")
        self.ICON_PATH = 'notepad_icon.png'
        os.makedirs(self.TARGET_DIR, exist_ok=True)