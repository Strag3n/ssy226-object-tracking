# Python pipeline

| File | Purpose |
| --- | --- |
| `start.py` | Run this file. Contains settings, data loading and the test loop. |
| `object_pose_estimator/observation_models.py` | Grounding DINO detection and SAM2 segmentation for recorded RGB frames. |
| `video.py` | Saves the test video and measures FPS/Hz. |
| `object_pose_estimator/offline_runner.py` | Offline runner for 3D preparation and positional Kalman replay. |
| `object_pose_estimator/` | Unmodified Python modules and license from the original ROS pose estimator. |

## Run

Use your own Python environment. From the project folder:

```powershell
python -m pip install -r python-pipeline/requirements.txt
python python-pipeline/start.py
```

In VS Code, select your environment's Python interpreter and run `start.py`. The first model download requires internet access.

## Settings

Edit the settings near the top of `start.py`:

```python
DATA_FOLDER = PROJECT_FOLDER / 'extracted-data'
RECORDING = None       # Choose one test from the terminal menu; or set 'apple-clutter'.
MAX_MESSAGES = None    # All RGB frames; use 10 for a short trial.
CONFIDENCE = 0.25
```

`CATEGORIES_BY_SCENARIO` selects the prompt for each recording (apple, banana, box or mug). `OBJECT_CATEGORIES` is the fallback. These are explicit text prompts, like the original pipeline's object_categories.

When started, the program lists the recordings. Enter the number of the test you want to run, or q to quit. Only that recording runs. You can also set RECORDING directly to a scenario name in the code.

## Results

Videos, annotated images and timing reports are saved under `../results/detection/<run>/<recording>` relative to this code folder.

- `detections.mp4`: boxes, confidence and measured detection speed.
- `frame_metrics.csv`: source timestamps and per-frame timings.
- `metrics.json`: per-recording measurements.
- Run-level `summary.csv`: comparison across recordings.

Source FPS is the recording's average image frequency. Detection Hz is processing speed excluding loading/export. End-to-end Hz includes loading, detection and saving. Playback follows source FPS, not processing speed. Model loading and warmup are excluded from measured rates. Detection fraction is not accuracy; there is no ground-truth evaluation.

## Original algorithm and tests

The original `_gdino_detect` and `_pick_device` functions are unchanged, including the original 320x240 image size and score/box calculations. The offline adapter replaces ROS transport and maps boxes to source-image size for drawing. Original copies, provenance and tests are in `../pipeline-reference`. License: AGPL-3.0, included in LICENSE.

Currently active: RGB object detection and video/performance output. SAM2, 3D association, pose estimation and tracking are later steps.

```powershell
python -m unittest discover -s pipeline-reference/tests -v
```


## Baseline configuration

MODEL_INPUT_SHORT_EDGE=None and CPU_THREADS=None preserve the original processor resolution and default threading. The original 320x240 callback resize remains, followed by the model processor's original resize. CONFIDENCE=0.25 matches the ObjectSegmenter constructor default (the separate original constant is 0.5 but is not used there). The copied detection and device-selection functions are checked verbatim by the reference tests.

Earlier runs using MODEL_INPUT_SHORT_EDGE=240 are reduced-resolution experiments, not original-preprocessing baseline results. Run a fresh test for the baseline and keep these outputs separate. Hardware, model/dependency revisions and the offline adapter still differ from the original ROS deployment, so numerical or speed equivalence has not been established.

## Test menu and outputs

Open `start.py` in VS Code, choose a recording, then select a test.

Every test records timing automatically. The summary in the terminal shows the stage time and the total processing and export time. Model load time is reported separately in the saved results where applicable.

| Option | Input | New output under `baseline/results/` |
| --- | --- | --- |
| 1. Object detection | Extracted RGB frames | `detection/<run>/<recording>/` with annotated images, detection JSON, video and timings |
| 2. Full pipeline | Extracted RGB frames, point clouds and camera calibration | `full-pipeline/<recording>/<run>/` with detections, `measurements.jsonl`, video and stage timings |
| 3. Segmentation only | Saved detections from 1 or 2 for the selected recording, plus its RGB frames | `segmentation/<recording>/<run>/` with `masks/`, `segments.jsonl` and a manifest |
| 4. Estimate 3D positions for Kalman | Saved masks from 3, source detections, and recorded images, clouds and calibration | `3d-measurements/<recording>/<run>/` with raw XYZ measurements, masks and a timing summary in `manifest.json` |
| 5. Kalman only | Saved 3D measurements from 4 or 2 | `kalman-only/<recording>/<run>/` with `kalman.csv`, `metrics.json` and `kalman.mp4` |

Use **1 ? 3 ? 4 ? 5** for separate stages. Use **2 ? 5** to replay Kalman on the raw 3D measurements of a full run. The menu lists saved inputs matching the selected recording, including compatible older runs. Only completed upstream runs are offered. Kalman replay loads no models. Its video uses the original RGB images and overlays the measured and filtered estimates.

Options 2 and 3 require SAM2. In the same Python environment as VS Code:

```powershell
$env:SAM2_BUILD_CUDA = '0'
python -m pip install "git+https://github.com/facebookresearch/sam2.git"
```

Option 4 combines each saved mask with depth and camera calibration to estimate a raw XYZ position. Those positions are the input to option 5. Keep the source detection files referenced by each segmentation run available.

### Baseline boundaries

This is the original **positional filter and positional measurement logic in an offline adapter**, not numerical equivalence with the full live ROS system. Its 5 Hz input schedule uses the latest saved detection and cloud available at each bag-time tick, without looking ahead. Fixed original filter dt is retained. Loss timeout uses recorded time instead of wall-clock inference time. Hardware/model/dependency revisions and actual ROS scheduling still differ. Cached detections alone cannot reproduce frames that a slow live ROS service would have skipped.

The recordings have no joint states; gripper-closed stays false, matching the original missing-joint behavior. Grasp/TF tracking and orientation filtering are outside this position-only test. The cloud must be organized and match the image dimensions. Like the original code, preparation indexes the cloud with the RGB mask and assumes registration. In the supplied apple recording, cloud and RGB optical-frame names differ. This emits a warning and is recorded in every cached row; registration is NOT verified and no coordinate transform is silently invented. The original assumption is retained for algorithm comparison, but physical 3D accuracy must not be claimed until registration is checked. Coordinates remain in the camera frame; robot/camera motion is not compensated. Invalid NaN/Inf positions are rejected rather than silently changing the original helper formulas.

Original helper/filter code is source-verified. Real SAM2/Depth Anything preparation has not been executed here because SAM2 is not installed. Pure cached-filter replay is tested independently; no synthetic results are presented as real tracking results.

`Segmenter.segment(rgb, detections)` in `object_pose_estimator/observation_models.py` returns the selected binary mask, its bounding box and SAM score. The offline runner calls this during 3D preparation. Kalman replay reads the saved measurements.

## Time all implemented stages in one test

Run start.py, choose one recording, then select **2. Full pipeline**. It runs detection, SAM2, 3D preparation and positional Kalman at the offline 5 Hz rate. `MAX_MESSAGES` limits source RGB frames before scheduling. `MAX_STEPS` in `object_pose_estimator/offline_runner.py` caps scheduled steps.

Results are under `baseline/results/full-pipeline/<recording>/<run>/`. Each run includes stage_timings.csv, timing_summary.json, measurements.jsonl and detections.mp4. CSV reports input loading, detection, SAM2, depth-model inference, other 3D/occlusion calculations, each positional Kalman call, export and per-step end-to-end time. Kalman mode identifies initialization, update, predict-only, waiting and reset. The depth execution flag shows when depth inference ran.

Model loading is timed separately. There is no hidden warmup or dropped measurement: the first full processed frame is saved as a cold-start measurement. JSON gives mean, median and p95 for all frames and separately excluding the first frame. GPU synchronization is used for timed model calls. Stage averages include genuine skipped stages as zero and should be read alongside depth_executed_frames and Kalman mode counts. Video Detection Hz describes the DINO stage only; the overlay lists other stage durations.

For baseline, use MODEL_INPUT_SHORT_EDGE=None. This measures the currently implemented offline position pipeline. It does not include the original rotation filter or grasp/TF handling, and it is not the real-time latest-frame experiment discussed earlier. Cloud registration remains an unverified original assumption. Full model execution is still to be verified on the user's installed models; tests can check orchestration with stub models independently.

## Visualize completed timing tests

Run `python python-pipeline/video.py`, choose a timed test (Enter selects latest). No models run. It saves timing_plots.png (stage means, per-step time allocation, Kalman call latency and processing-versus-source time) and position_plots.png when 3D estimates are available. The position plot is a measured-versus-filtered comparison in the camera frame, not accuracy against ground truth. Install matplotlib from requirements.txt if needed. All-step plots include cold start; timing_summary.json separately reports statistics excluding the first frame.
