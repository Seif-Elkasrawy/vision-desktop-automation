# Vision-Based Desktop Automation with Prompt Chaining Grounding

A Python-based desktop automation agent that locates and interacts with desktop UI elements dynamically using a multi-stage AI vision pipeline, with multiple fallback layers for reliability.

## How It Works

The orchestrator fetches posts from an API, locates the Notepad icon on the desktop using visual grounding, launches Notepad, types and saves each post as a `.txt` file, and stores an annotated screenshot of where the icon was found.

```
Fetch posts → Ground Notepad icon → Click → Type content → Save file → Validate launch
```

## Vision Pipeline: Prompt Chaining

The core of this project is a three-stage prompt chaining architecture. Rather than asking one AI prompt to locate an icon on the full screen, the task is broken into three focused stages where each stage's output feeds into the next.

### Stage A — Planner
Receives the full desktop screenshot and scans all quadrants to return up to 3 ranked candidate areas where the target icon might be, each with a neighboring icon named for spatial context.

### Stage B — Grounder
Receives each candidate crop from the Planner and locates the icon precisely within that smaller region. Working on a zoomed-in crop rather than the full screen makes this step significantly more accurate.

### Stage C — Verifier
Receives the predicted pixel coordinates from the Grounder. A red circle is drawn at the predicted point and Gemini is asked to confirm whether the circle is centred on the correct icon graphic (not the text label, not an adjacent icon). If the circle is off-centre, the Verifier returns a corrected bounding box.

## Fallback Chain

If any stage of the pipeline fails, the system falls back in order:

1. **Gemini** (Planner → Grounder → Verifier) — primary pipeline. Each individual Gemini API call is retried up to 4 times with exponential backoff on transient errors (503, 429, RESOURCE_EXHAUSTED), doubling the wait between attempts.
2. **GPT-4o** — a single prompt sends the full screenshot to OpenAI and asks it to return the icon's pixel coordinates directly.
3. **OpenCV template matching** — fully local, no AI or network required. Matches a saved `notepad_icon.png` template against the live screenshot using pixel similarity scoring.

The entire fallback chain is retried up to 3 times, with a fresh screenshot taken on each retry.

## Coordinate Caching

Once the target icon is successfully grounded, its coordinates are cached and reused for all subsequent posts, skipping the entire AI pipeline. If a click using cached coordinates fails to open Notepad (validated by `WindowManager` polling for the window title), the cache is invalidated and the full pipeline runs again to find the updated position.

```
First post:        Gemini pipeline → cache coords → click → validate ✓
Subsequent posts:  use cache → click → validate ✓
Cache invalid:     invalidate → Gemini pipeline → cache new coords → retry
```

## Deliverables

For each post processed, the following are written to `~/Desktop/tjm-project/`:

- `post_{id}.txt` — the post content typed into Notepad
- `screenshot_post_{id}.png` — annotated screenshot with a red circle and crosshair marking where the icon was found

## Project Structure

```
main.py           — Orchestrator: runs the post loop, manages caching and retries
vision.py         — VisionManager: Gemini pipeline, OpenAI fallback, OpenCV fallback
prompts.py        — All prompt templates and JSON schemas for Planner, Grounder, Verifier
actions.py        — AutomationActions: mouse/keyboard control via pyautogui
window_manager.py — WindowManager: validates Notepad launched successfully
data_client.py    — DataClient: fetches posts from the placeholder API
config.py         — Config: loads API keys from .env, sets output paths
constants.py      — Shared timing constants and radius values
utils.py          — save_annotated_screenshot helper
```

## Requirements

- `uv` for dependency management (dependencies declared inline in `main.py`)
- A `.env` file with `GEMINI_API_KEY` and `OPENAI_API_KEY`
- `notepad_icon.png` in the project root (used by the OpenCV fallback)
- A Notepad shortcut visible on the Windows desktop