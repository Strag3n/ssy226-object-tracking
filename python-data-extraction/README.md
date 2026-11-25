# Python data extraction

## Where does pose-estimation-data-new fit in?

`pose-estimation-data-new` is the **input data folder**. It contains the original recordings. `python-data-extraction` contains the **code** that reads them. `extracted-data` contains the **output** produced when you run the code.

| Folder | Purpose | Example file |
| --- | --- | --- |
| `python-data-extraction` | Python code and instructions | `extract.py` |
| `pose-estimation-data-new` | Original recorded measurements | `apple-clutter/apple-clutter_0.db3` |
| `extracted-data` | Exported data created by the script | `apple-clutter/apple-clutter_0/index.csv` |

All three folders sit directly inside `Design project SSY226`.

## Set the paths directly in extract.py

Edit the settings near the top of `extract.py`:

```python
PROJECT_FOLDER = Path(__file__).resolve().parent.parent
INPUT_FOLDER = PROJECT_FOLDER.parent / 'pose-estimation-data-new'
OUTPUT_FOLDER = PROJECT_FOLDER / 'extracted-data'
LIMIT_PER_TOPIC = None
```

`PROJECT_FOLDER` is found from the script location. `INPUT_FOLDER` points to the original recordings. `OUTPUT_FOLDER` selects where exports are saved. No command-line arguments are needed.

You can also enter a full Windows path directly:

```python
INPUT_FOLDER = Path(r'C:\your-project\pose-estimation-data-new')
```

Install dependencies once, then run the script from the project root (or use your editor's Run button):

```powershell
python -m pip install -r python-data-extraction/requirements.txt
python python-data-extraction/extract.py
```

Python 3.10+, NumPy and Pillow are required. ROS, Docker and Linux are not required. Use the full path to your Python executable if `python` is unavailable.

The existing `extracted-data` folder is already populated. The script refuses to overwrite an existing output folder. Change `OUTPUT_FOLDER` to a new name before another run. For a small trial, set `LIMIT_PER_TOPIC = 1` and choose a new output name such as `extraction-preview`. Set it back to `None` for all messages. A trial is not a synchronized set of frames.

## Follow one recording from input to output

For example, the script reads this original file:

`pose-estimation-data-new/apple-clutter/apple-clutter_0.db3`

Here is the exact connection between the settings and the code:

1. `INPUT_FOLDER` near the top of the code points to `pose-estimation-data-new`.
2. The last line calls `extract(source=INPUT_FOLDER, target=OUTPUT_FOLDER, limit=LIMIT_PER_TOPIC)`.
3. Inside `extract()`, `source.rglob('*.db3')` searches that input folder recursively. This finds the database above, as well as the recordings in the other subfolders.
4. `sqlite3.connect(... + '?mode=ro', uri=True)` opens each discovered database read-only.
5. SQL reads the `topics` table to identify each stream, and the `messages` table to get each message's topic ID, recording timestamp and binary data.
6. `decode(data, kind)` unpacks the recorded binary CDR message into named fields.
7. `save_message()` chooses the RGB, depth, point cloud, calibration or transform exporter.
8. The exporter writes the result under `extracted-data/apple-clutter/apple-clutter_0/`. `index.csv` records which topic produced each file.

The script prints the resolved input path, the number of discovered databases, the output path, and each database as it is read. This lets you see which folder is being used.

The `.db3` files contain recorded ROS messages, not ready-made PNG images. The existing `metadata.yaml` files refer to older database names, so this script discovers the actual `.db3` files directly instead of using those names.

## Which stream produces which output?

These are the streams in the project's recordings. The dispatcher uses the ROS message type and image encoding. Topic IDs can vary between databases, so use `index.csv` to identify the data inside `topic_001`, `topic_002`, etc.

| Recorded topic | Export function | Output |
| --- | --- | --- |
| `/head_front_camera/rgb/image_raw` | `extract_rgb()` | RGB PNG + metadata JSON |
| `/head_front_camera/depth/image_raw` | `extract_depth()` | Raw depth NPY + metadata JSON |
| `/head_front_camera/depth/rgb/points` | `extract_point_cloud()` | Structured point cloud NPY + metadata JSON |
| `/head_front_camera/rgb/camera_info` | `extract_camera_info()` | Calibration JSON |
| `/tf` | `extract_transforms()` | Coordinate transform JSON |

### RGB extraction

`decode()` reads the image header, dimensions, encoding, byte order, row length (`step`) and pixel buffer. `image_array()` interprets the buffer using these properties, including any row padding.

For `rgb8`, the result is a `uint8` array with shape `(height, width, 3)`. `extract_rgb()` saves it as a lossless PNG. `bgr8` is reordered to RGB; `mono8` is saved as grayscale. There is no resizing, cropping or normalization.

### Depth extraction

Depth and RGB both use the ROS `Image` message type. Their `encoding` values specify how to interpret their pixels. `16UC1` means one unsigned 16-bit channel; `32FC1` means one 32-bit floating-point channel.

`image_array()` builds an `(H, W)` array. `extract_depth()` saves it as NPY, preserving the original values and data type, including zeros and NaNs. The encoding specifies the data type, not the physical unit. Verify the camera configuration before converting values to meters. No unit conversion or depth visualization is performed here.

### Point cloud extraction

**The point cloud is already recorded in the database. This script reads it directly; it does not generate it from the depth image.**

`decode()` reads the PointCloud2 message, including:

- `fields`: each field's name, byte offset, data type and number of values.
- `point_step`: bytes per point.
- `row_step`: bytes per row.
- `is_bigendian`: byte order.
- `data`: the point buffer itself.

`cloud_array()` uses these definitions to build a structured NumPy array with shape `(H, W)`. `extract_point_cloud()` saves it as NPY. All recorded fields are retained, including any color fields. `cloud['x']`, `cloud['y']` and `cloud['z']` give coordinates when those fields exist.

NaNs, point organization and coordinate frame are preserved. No points are removed and no transform is applied. The coordinate frame is in the JSON field `frame_id`. A matching row/column in RGB and the cloud must not be assumed to represent the same ray without verified registration.

### Camera calibration

`extract_camera_info()` writes the decoded CameraInfo fields to JSON: image dimensions, distortion model and coefficients (`D`), camera matrix (`K`, 3 by 3), rectification matrix (`R`, 3 by 3), projection matrix (`P`, 3 by 4), binning and ROI. Matrices are flat lists in row-major order.

### Coordinate transforms

`extract_transforms()` writes each recorded transform to JSON: timestamp, parent frame (`frame_id`), child frame (`child_frame_id`), translation `[x, y, z]` and quaternion `[x, y, z, w]`. The transforms are exported but not applied to the cloud.

## Find and load exported data

Every database output folder contains:

- `index.csv`: one row per exported message, with topic, type, message ID, recording timestamp, header timestamp, frame ID and relative data file path.
- `summary.json`: source database, original topic counts and extracted counts.
- `topic_<id>` folders: individual PNG, NPY and JSON files named `<recording_timestamp_ns>_<message_id>`.

The top-level output folder also has `summary.json` covering all databases. Timestamps are integer nanoseconds. TF has an individual timestamp per transform in JSON, so its CSV header timestamp and frame columns are empty.

This runnable example loads the first export from each camera stream in apple-clutter:

```python
import csv
from pathlib import Path

import numpy as np
from PIL import Image

folder = Path('extracted-data/apple-clutter/apple-clutter_0')
with (folder / 'index.csv').open(encoding='utf-8', newline='') as handle:
    rows = list(csv.DictReader(handle))


def first_file(topic):
    row = next(row for row in rows if row['topic'] == topic)
    return folder / row['file']


rgb = np.asarray(Image.open(first_file('/head_front_camera/rgb/image_raw')))
depth = np.load(first_file('/head_front_camera/depth/image_raw'), allow_pickle=False)
cloud = np.load(first_file('/head_front_camera/depth/rgb/points'), allow_pickle=False)

print('RGB:', rgb.shape, rgb.dtype)
print('Depth:', depth.shape, depth.dtype)
print('Cloud:', cloud.shape, cloud.dtype.names)
xyz = np.stack([cloud['x'], cloud['y'], cloud['z']], axis=-1)
valid_points = xyz[np.isfinite(xyz).all(axis=-1)]  # Optional analysis step.
```

These are the first messages from each stream, not a synchronized triplet. RGB, depth and clouds keep their independent timestamps. Use header times and an explicit tolerance in a subsequent synchronization step. No segmentation, object poses or ground truth are generated.

## Supported formats and checks

The reader supports plain CDR1 serialization and the four ROS message types listed above (RGB and depth share Image). Unsupported types or formats stop the run with an error. Inputs are opened read-only. There is no automatic resume; if interrupted, choose a new output directory for a fresh run.

Run the existing tests from the project root:

```powershell
python -m unittest discover -s python-data-extraction -v
```
