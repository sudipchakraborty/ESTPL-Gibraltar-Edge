# Gibraltar Edge

Start the desktop application with `run_edge.bat` (uses the project virtual environment).

## Inspection workflow

- **Camera 1:** OCR text inspection, with local contrast enhancement for embossed lettering. Show readable reference text inside the ROI and click **Save**. The captured image and recognized reference text are stored in that camera's repository folder.
- **Cameras 2 and 3:** Object inspection. Draw/save an ROI around the inspection area, place **one object** in it, and click **Save**. Save stores the exact captured ROI image plus the detected class and its appearance feature. References are independent for each camera and are loaded on restart. No separate Train Model step is needed.
- Stable placement starts the observation period. At its end, the best confident reading determines green tick/match or red cross/mismatch. Only one event and enabled sound occur per placement. Brief glitches and candidates that disappear during observation do not trigger errors.
- The result stays latched until five consecutive absence readings span at least 1.5 seconds, then the camera returns to WATCHING.
- Use **Audio Alert Config** to enable each camera and its match/mismatch sounds. Existing enabled/disabled settings are preserved.

Object references compare the detected class and a colour/appearance histogram from the object's unannotated crop. They do not retrain YOLO or add new object classes. Objects must be recognized by the installed `models/weights/yolov8n.pt` detector. For objects it cannot detect, a suitable trained detector is needed. Use a tight ROI to exclude unrelated objects. Multiple detected objects do not pass a single-object inspection.

## Configuration

Edit `Projects/config.json` and restart the app. Each camera has its own section:
`camera_1_inspection`, `camera_2_inspection`, and `camera_3_inspection`.

| Setting | Meaning | Default |
| --- | --- | --- |
| `placement_wait_seconds` | Observation time after stable placement is confirmed | 2.0 |
| `match_score_threshold` | Required text similarity or object appearance score | 0.85 |
| `ocr_confidence_threshold` | Minimum OCR confidence (Camera 1) | 0.85 |
| `object_confidence_threshold` | Minimum object detector confidence (Cameras 2/3) | 0.60 |
| `placement_confirm_readings` | Consecutive consistent readings before observation | 3 |
| `placement_confirm_seconds` | Minimum duration of placement confirmation | 0.6 |
| `placement_min_text_length` | Minimum normalized text length (Camera 1); leave at 1 for object cameras | 2 / 1 |

Scores use 0..1 (0.85 means 85%). Timings follow completed inference readings, so slow inference can extend the elapsed time. Camera 1 infers placement/removal from readable text; object cameras use detector presence/absence. These are not physical presence sensors.

## Validation

Run `.venv\Scripts\python.exe -m unittest testing.test_inspection_flow -v` for offscreen flow checks. They cover Save, ROI crops, camera isolation, timing, match/mismatch, removal, glitches, and object event metadata without cameras or backend writes. Physical camera accuracy and audible playback must be verified on the installed system.
