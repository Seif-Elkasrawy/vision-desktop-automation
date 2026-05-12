# Vision-Based Desktop Automation with Robust Local Grounding

A Python-based desktop automation agent designed to locate and interact with UI elements dynamically. 

## Approach: Hybrid Visual Grounding
This project explores the "ScreenSeekeR" framework (from *ScreenSpot-Pro*) but is optimized for local reliability and high availability.

### 1. Primary Strategy (Agentic Grounding)
The codebase includes a conceptual implementation of the **ScreenSeekeR cascaded search** methodology:
- Utilizes the "Planner/Grounder" architecture to perform semantic UI positioning.
- Designed to handle complex, high-resolution professional environments by zooming in on candidate regions.

### 2. Operational Strategy (Local Robust Fallback)
Per Section 5 of the requirements (**Graceful Degradation**), the system is configured to prioritize a local OpenCV engine. This ensures:
- **Low Latency**: Faster icon detection than network-dependent APIs.
- **Reliability**: Automation continues even if cloud AI quotas are exceeded or internet is unstable.
- **Privacy**: Processing occurs locally on the machine.

## Technical Implementation Highlights
- **Error Handling**: Implements retry logic and window state validation.
- **Automation Flow**: Dynamic screenshotting -> Icon Grounding -> Double-click launch -> Content Processing -> File Management.
- **Deliverables**: Annotated screenshots available in the `tjm-project` folder on the Desktop.

## Requirements
- `uv` (Fast dependency management)
- `pyautogui` for peripheral control
- `OpenCV` for local vision grounding
- `Notepad` shortcut on the desktop