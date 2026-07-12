# Air Canvas ✋🎨

Draw in the air using just your index finger — no mouse, no touchscreen, just your webcam and your hand.

Built as a self-directed learning project to understand real-time computer vision, hand-tracking, and gesture-based interaction using Python.

## What it does

- Tracks your hand live through your webcam using Google's MediaPipe Hand Landmarker
- Raise your **index finger only** to draw on screen
- Show **two fingers** to pause drawing and move without leaving a trail
- Make a **fist** to save your drawing as an image
- Show an **open palm** to capture a snapshot of the live camera feed

## Gesture Map

| Gesture | Fingers Up | Action |
|---|---|---|
| ☝️ Index finger only | 1 | Draw |
| ✌️ Two fingers | 2 | Pause / move without drawing |
| ✊ Fist | 0 | Save drawing as image |
| ✋ Open palm | 4 | Take a snapshot |

## Tech Stack

- **Python**
- **OpenCV** — webcam capture, drawing, image saving
- **MediaPipe** — real-time hand landmark detection (21 points per hand)
- **NumPy** — distance calculations for smooth line interpolation

## How it works

MediaPipe's Hand Landmarker returns 21 (x, y, z) coordinate points per detected hand every frame. This project doesn't use a trained gesture classifier — instead, it compares each fingertip's position to its knuckle joint to determine which fingers are extended, then maps the resulting finger count to an action.

To keep drawing feel smooth despite frame-to-frame detection noise:
- Fingertip position is smoothed using exponential averaging instead of raw coordinates
- Line segments are interpolated between frames to avoid gaps during fast movement
- Drawing state uses an instant-start/delayed-stop pattern to avoid losing the beginning of a stroke while still tolerating brief detection dropouts

## Installation

```bash
git clone https://github.com/udaykishorekota-sketch/Air-Canvas.git
cd Air-Canvas
python -m venv venv
venv\Scripts\activate      # Windows
pip install opencv-python mediapipe pyautogui numpy
```

Download the hand landmark model:
```bash
mkdir models
```
Then download `hand_landmarker.task` from [Google's MediaPipe model zoo](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) into the `models/` folder.

## Run

```bash
python main.py
```

Press **ESC** (with the window focused) to quit.

## Known Limitations

- Fine cursive handwriting is imperfect — webcam frame rate and hand-detection confidence create a real trade-off between responsiveness and false positives. The tool is tuned for **deliberate, moderate-speed strokes** rather than fast cursive signatures.
- Currently hardcoded for right-hand use; left-hand gesture logic is not yet implemented.
- Single-hand tracking only.

## Future Improvements

- Color palette for switching pen colors mid-drawing
- Left-hand support
- Straight-line drawing mode (toggle via gesture)
- Thumb-inclusive gestures using distance-based detection instead of axis comparison, for more gesture variety

## Why I built this

I wanted a hands-on project to genuinely understand computer vision fundamentals — not just follow a tutorial, but debug real problems like detection jitter, frame-rate trade-offs, and gesture stability. This project reflects that iterative process rather than a polished, flawless first draft.