# SSY226 — Offline Object Tracking Baseline

This project evaluates an offline object detection and position-tracking pipeline using recorded robot camera data. It supports object detection, segmentation, depth estimation, 3D position and occlusion estimation, positional Kalman filtering, manual 2D reference annotation, and result plots.

The recordings are processed offline. The pipeline does not control the robot. The implemented full-pipeline test covers position tracking; rotation filtering and grasp/TF handling are outside this test.

## Quick start

All commands below assume your terminal is inside `baseline` and your Python environment is activated.

```powershell
python python-pipeline/start.py
```

Choose a recording, then a test type. To annotate reference points or plot a completed full-pipeline test:

```powershell
python reference_points.py
python plot_test_results.py
```

The annotation tool opens a file picker for an extracted recording's `index.csv`. The plotting tool opens a folder picker for a saved test run.

## Project layout

| Path | Purpose |
| --- | --- |
| `../pose-estimation-data-new/` | Original recordings (`.db3` files), outside `baseline`. |
| `python-data-extraction/` | Code for exporting the recordings without ROS. |
| `extracted-data/` | Exported images, point clouds, calibration, transforms and per-recording indexes. |
| `python-pipeline/start.py` | Entry point for the five test modes and main detection settings. |
| `python-pipeline/object_pose_estimator/` | Observation models, offline runners, positional filter and related modules. |
| `python-pipeline/video.py` | Video export, timing utilities and additional plots. |
| `reference_points.py` | Interactive object-centre annotation tool. |
| `true-data/` | Saved manual annotations, organised by recording. |
| `plot_test_results.py` | Folder picker and combined position/timing plotting entry point. |
| `plot_timing.py` | Timing plotting helper called by `plot_test_results.py`. |
| `results/` | Saved test runs and their plots. |

## Python environment and installation

The code uses the Python environment selected when you launch it. It does not load a particular Conda environment internally. In VS Code, select the intended interpreter using **Python: Select Interpreter**.

If you already use the `design-course` environment:

```powershell
conda activate design-course
```

For a new installation, Python 3.11 is a suitable starting point:

```powershell
conda create -n ssy226 python=3.11
conda activate ssy226
python -m pip install -r python-pipeline/requirements.txt
python -m pip install -r python-data-extraction/requirements.txt
```

The pipeline requirements include NumPy, Pillow, PyTorch, torchvision, Transformers, OpenCV and Matplotlib. Interactive file pickers and annotation require Tkinter, which is included in many Windows Python/Conda installations.

Segmentation and full-pipeline/3D preparation also require the separate SAM2 package. Git is required for this installation command. For a Windows CPU installation attempt in PowerShell:

```powershell
$env:SAM2_BUILD_CUDA = '0'
python -m pip install 'git+https://github.com/facebookresearch/sam2.git'
```

A working SAM2 installation must be checked on the target machine. This command alone does not establish Windows or GPU compatibility. Choose PyTorch/CUDA packages appropriate to your machine if using a GPU. The first model use needs internet access to download weights; later runs can reuse cached downloads.

The requirements contain version ranges, not a locked environment. Record the actual dependency versions, device, model revisions and settings when comparing experiments.

Do not commit a complete Python/Conda environment to Git. Each collaborator creates their own environment and installs dependencies. Share large recordings separately and place them at the documented paths.

## Exporting the recordings

Skip this step if `extracted-data` already contains the recordings you need.

Check the settings at the top of `python-data-extraction/extract.py`:

```python
INPUT_FOLDER = PROJECT_FOLDER.parent / 'pose-estimation-data-new'
OUTPUT_FOLDER = PROJECT_FOLDER / 'extracted-data'
LIMIT_PER_TOPIC = None
```

Then run:

```powershell
python python-data-extraction/extract.py
```

The script reads the original databases without ROS and exports each recording to, for example, `extracted-data/apple-clutter/apple-clutter_0/`. Its `index.csv` links saved files to topics, timestamps and message IDs. Original recordings are opened read-only.

The extractor refuses to overwrite an existing output folder. Use a new `OUTPUT_FOLDER`, or intentionally remove the old export before extracting again. If you change the output location, also update the pipeline's `DATA_FOLDER`.

Recorded depth images and point clouds are sensor data. The depth-model output generated during a pipeline test is a separate product; do not treat them as the same measurement.

## Running tests

```powershell
python python-pipeline/start.py
```

First choose the recording, then choose a test mode:

| Mode | Test | Required input |
| --- | --- | --- |
| 1 | Object detection only | Extracted RGB frames. |
| 2 | Full pipeline | Extracted RGB, organised point clouds and camera calibration; detection, SAM2 and depth models. |
| 3 | Segmentation only | Saved detections from mode 1 or 2, matching extracted frames and SAM2. |
| 4 | 3D position preparation | Saved segmentation from mode 3, matching detections, extracted point clouds/calibration and required models. |
| 5 | Kalman only | Saved 3D measurements from mode 4 or a completed mode 2 run. No detection, SAM2 or depth inference is rerun. |

For modes 3–5, the menu asks you to select saved input for the same recording. Separate-stage tests create their own run folders; they do not overwrite the source run.

The full-pipeline run saves masks and depth arrays, but the segmentation-only `segments.jsonl`/manifest interface used by mode 4 is produced by mode 3.

### Main settings

Edit these near the top of `python-pipeline/start.py`:

| Setting | Meaning |
| --- | --- |
| `DATA_FOLDER` | Root of the extracted recordings. |
| `RECORDING = None` | Show the recording menu. A recording/scenario string can select one directly. |
| `MAX_MESSAGES = None` | Use all source RGB frames. An integer limits source frames before full-pipeline scheduling. |
| `CONFIDENCE = 0.25` | Detection confidence threshold. |
| `DEVICE = None` | Automatic device selection for the detector. |
| `MODEL_INPUT_SHORT_EDGE = None` | Preserve the original detector processor resolution; keep this for baseline comparisons. |
| `CPU_THREADS = None` | Keep PyTorch's default CPU threading. |
| `CATEGORIES_BY_SCENARIO` | Text prompts for each scenario: apple, banana, box or mug. |
| `OUTPUT_FOLDER` | Root of saved results. |
| `WARMUP_FRAMES = 1` | Extra unmeasured inference for the detection-only test. |

Separate-stage settings, including `DEVICE`, `MAX_STEPS` and `REQUIRE_BASELINE`, are in `object_pose_estimator/offline_runner.py`. These are separate from `start.py` settings.

### Why 250 source frames can become approximately 50 steps

Detection-only mode processes all selected RGB frames. The full pipeline uses recorded timestamps to schedule steps at `ESTIMATION_RATE_HZ = 5.0`, taking the latest available RGB frame at each 0.2-second tick. Separate segmentation/3D preparation also use this scheduling logic.

For example, 10 seconds of video at 25 FPS contains 250 source frames but approximately 50 scheduled full-pipeline steps. Steps lacking required point cloud/calibration data may be skipped. This does not delete source images.

The scheduled rate is not measured processing speed. A step can take several seconds on a CPU even though its recording-time interval is 0.2 seconds. Keep the sampling frequency and Kalman time step consistent when changing experiments.

## Saved result structure

Each new test gets a separate run folder:

`results/<scenario>/<recording>/<timestamp>_<id>_<test-type>/`

For example:

`results/apple-clutter/apple-clutter_0/20261125_150313_a135e8ca_full-pipeline/`

A new full-pipeline run contains:

| Folder | Contents |
| --- | --- |
| `object-detection/` | Per-frame detection JSON and annotated PNG images. |
| `segmentation/` | Binary masks (`.npy`) when segmentation succeeds. |
| `depth-model/` | Model depth arrays (`.npy`) when depth inference executes. |
| `3d-occlusion/` | Per-step raw position, depth selection and occlusion-state JSON. |
| `kalman/` | `measurements.jsonl`: raw and filtered positions, calibration, states, detections and source-image references. |
| `timing/` | `stage_timings.csv` and `timing_summary.json`. |
| `video/` | `pipeline.mp4` and `frame_metrics.csv`. |
| `plots/` | `position_comparison.png` and `processing_time.png`, created by the plotting script. |

Frame files use timestamps and message IDs to identify their source. Stage inference can be skipped, so not every folder necessarily has a file for every scheduled step.

Detection-only runs save detection images, JSON, video and metrics in `object-detection/`, with run summaries in the run folder. Standalone segmentation, 3D and Kalman tests save their files in `segmentation/`, `3d-measurements/` or `kalman-only/` within their own runs.

Older runs may lack masks or depth arrays because previous versions did not save them. Moving an older run into the new structure cannot recreate missing model outputs.

## Manual 2D reference annotations

```powershell
python reference_points.py
```

Select `index.csv` from the desired extracted recording.

| Control | Action |
| --- | --- |
| Left click on the image | Mark the object's centre as visible and advance. |
| O | Mark the object as occluded and advance. |
| A | Mark it as absent and advance. |
| Left/right arrow | Move between frames. |
| Space | Play/pause. |
| Clear frame | Remove the current annotation. |
| Clear all | Confirm removal of all annotations for the open recording and return to the first frame. |

Annotations save immediately to `true-data/<scenario>/<recording>/reference_points.csv`. Reopening the same recording loads existing annotations. Coordinates `u_px` and `v_px` refer to the original image, with the origin at the top-left. Hidden/absent annotations have empty coordinates.

Use a consistent definition of the object's centre. Annotate densely around occlusion boundaries if evaluating recovery. Match results to annotations within the same recording using message IDs and timestamps.

These annotations provide a manual 2D reference. They do not establish true depth, 3D position or rotation, and they include human annotation uncertainty.

## Plotting a completed test

```powershell
python plot_test_results.py
```

Choose the complete run folder, such as `20261125_150313_a135e8ca_full-pipeline`. The script finds the saved detections, matching reference annotations, Kalman measurements and timing data. If the expected reference CSV is missing, it asks you to select one. A folder containing several runs is rejected; select one run instead.

The selected category is inferred from the most frequent saved detection label. The command-line `--category` option can specify a category for manual, noninteractive use.

### Position comparison

`plots/position_comparison.png` shows image coordinates and 2D pixel errors for:

- The manually annotated reference centre.
- The highest-confidence matching detection's bounding-box centre.
- The raw 3D position projected into the RGB image, when available.
- The filtered 3D position projected into the RGB image, when available.

The background shows saved pipeline visibility states, or manual visibility when pipeline states are unavailable. The raw-versus-filtered error comparison uses shared annotated frames where both estimates are valid. It does not interpolate missing detections or annotations.

### Processing time

`plots/processing_time.png` shows stacked coloured areas for exactly five stages:

1. Object detection.
2. Segmentation.
3. Depth model.
4. 3D position and occlusion.
5. Kalman filter.

The x-axis is processing step; the y-axis is seconds per step. Band thickness represents that stage's duration. The total line, mean, median and 95th percentile use the sum of these five stages. Loading, saving, other overhead and model initialisation are excluded from this graph. Skipped stages contribute zero. All saved steps are included, including the first processed frame if it is present in the saved data.

Running the plotter again overwrites these two named images for the same test run. Other runs retain their own plots. If a run has no timing data, only the position comparison is created. No models are rerun to create the plots.

## Interpreting measurements

| Measurement | Interpretation |
| --- | --- |
| Detection fraction | Fraction of frames with at least one detection; this is not detection accuracy. |
| 2D centre error | Distance in pixels from the manual reference, only on matching annotated frames. |
| Detection Hz | Measured speed of the detection stage. |
| End-to-end Hz/time | Includes additional work such as loading and exporting, as defined in the saved test metrics. |
| Five-stage timing total | Sum of the five compute stages displayed in the timing graph. |
| Recovery after occlusion | Requires a defined first-visible frame and a criterion for successful tracking; not automatically computed by these plots. |

Compare like-for-like runs: same recording, selected frames, prompts, preprocessing, model versions and hardware. Detection-only mode uses a warmup; full-pipeline timing reports model loading separately and does not use that detection-only warmup.

The offline adapter retains original positional logic but is not numerical or timing equivalence with a live ROS deployment. RGB/cloud registration is an unverified assumption: different optical-frame names can trigger a warning, and the code does not invent a coordinate transform. Coordinates remain in the camera frame; robot/camera motion is not compensated. Do not interpret these results as validated physical 3D accuracy.

## Cleaning up and troubleshooting

| Situation | What to check |
| --- | --- |
| Start over with test results | Remove the selected run or `results`. This deletes results/caches, not code; rerun tests to regenerate them. |
| Keep annotations | Preserve `true-data`; deleting it loses manual work. |
| Reuse recordings without extracting again | Preserve `extracted-data`. |
| No recordings found | Check `DATA_FOLDER` and the presence of per-recording `index.csv` files. |
| No saved input in modes 3–5 | Run the prerequisite test for that recording and ensure it completed. |
| SAM2 import error | Install SAM2 into the same interpreter that runs the script. |
| No detections found by the plotter | Select one completed run containing per-frame detection JSON. |
| No matching reference frames | Use the reference CSV from the same recording; check message IDs and timestamps. |
| Fewer full-pipeline steps than RGB frames | Check 5 Hz scheduling, `MAX_MESSAGES`, `MAX_STEPS` and missing cloud/calibration data. |

For sharing via Git, include code, dependency files, documentation and the relevant licenses. Keep Python environments, generated results, downloaded model weights and large datasets out of Git. Share annotations if appropriate for the project. Preserve the included license files and upstream attribution when distributing copied source.


## Pipeline video overlays and rebuilding

Full-pipeline tests automatically save `video/pipeline.mp4` with blue detection boxes, a green SAM2 mask at 28% opacity with a bright green contour, an orange raw 3D measurement and a yellow Kalman crosshair/ring. Positions are projected into the RGB image using the saved calibration. A separate header above the complete source image shows visibility state, Kalman operation and stage durations. Missing masks or estimates are not fabricated; off-image estimates cannot be drawn. A predicted Kalman position can remain visible during occlusion even when no segmentation mask is available.

To rebuild videos from saved results without rerunning detection, depth or tracking:

```powershell
python rebuild_pipeline_videos.py --all
```

To also recompute masks that were not saved by older versions:

```powershell
python rebuild_pipeline_videos.py --all --recompute-missing
```

This optional step requires SAM2 and its model weights. Recomputed masks are stored in `segmentation-reconstructed` with provenance information; they are visualisation outputs, not the original timed masks. Rebuilding preserves source results and timing CSV files. It writes and validates a temporary video before replacing `pipeline.mp4`; older `detections.mp4` exports are retained.
