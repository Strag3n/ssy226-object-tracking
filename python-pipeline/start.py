"""START HERE: edit the settings below, then run this file.

Reads RGB images -> detects objects -> saves videos and FPS/Hz reports.
"""
import sys
sys.dont_write_bytecode = True

from collections import Counter
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from PIL import Image
from datetime import datetime
import json
import csv
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import os

from object_pose_estimator.observation_models import ObjectDetector
from video import TestVideo, source_timing, performance_summary


# ==================== EDIT YOUR SETTINGS HERE ====================
PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_FOLDER / 'extracted-data'
RECORDING = None  # None = choose ONE test from a menu; or set 'apple-clutter'.
MAX_MESSAGES = None  # None = all RGB frames; or e.g. 10 for a quick trial.
RGB_TOPIC = '/head_front_camera/rgb/image_raw'
GDINO_MODEL = 'IDEA-Research/grounding-dino-tiny'
CONFIDENCE = 0.25  # Original ObjectSegmenter constructor default.
DEVICE = None  # Original automatic selection: CUDA, MPS, CPU.
# BASELINE: preserve the original model processor resolution.
# Do not change this when comparing against the original pipeline.
MODEL_INPUT_SHORT_EDGE = None
CPU_THREADS = None  # Keep PyTorch default; optionally benchmark 2, 4 or 8.
OBJECT_CATEGORIES = ['apple']  # Fallback for scenarios not listed below.
# Explicit prompts for each test, like the original request.object_categories.
CATEGORIES_BY_SCENARIO = {
    'apple-clutter': ['apple'],
    'apple-moving-robot': ['apple'],
    'apple-occlusion': ['apple'],
    'banana-moving-robot': ['banana'],
    'banana-rotation': ['banana'],
    'big-box-moving-arm': ['box'],
    'box-moving-arm': ['box'],
    'box-rotation': ['box'],
    'mug-clutter': ['mug'],
    'mug-occlusion': ['mug'],
    'mug-rotation': ['mug'],
}
OUTPUT_FOLDER = PROJECT_FOLDER / 'results'
WARMUP_FRAMES = 1  # Extra inference before timing each recording; not video frames.
OPEN_VIDEO_WHEN_DONE = True  # Open the first completed video on Windows.
# ================================================================


# DATA READING: the exported index and its PNG/NPY/JSON files.
@dataclass(frozen=True)
class Measurement:
    recording: str
    message_id: int
    topic: str
    message_type: str
    bag_timestamp_ns: int
    header_timestamp_ns: int | None
    frame_id: str
    path: Path
    kind: str
    data: np.ndarray | dict
    metadata: dict
    # None means detection has not run; an empty tuple means no objects found.
    detections: tuple[dict, ...] | None = None


class Recording:
    """One exported database: its index and lazily loaded data files.

    Paths inside index.csv are relative to the index's parent folder.
    No dependency on the original ROS repository or extraction code.
    """
    def __init__(self, index_path: Path, name: str):
        self.index_path = index_path
        self.name = name
        with index_path.open(encoding='utf-8', newline='') as handle:
            self.rows = sorted(csv.DictReader(handle),
                               key=lambda row: (int(row['bag_timestamp_ns']), int(row['message_id'])))

    def __len__(self):
        return len(self.rows)

    def load(self, row: dict) -> Measurement:
        """Load a single PNG, NPY or JSON and its metadata.

        RGB is an (H,W,3) uint8 array; depth is a raw (H,W) array.
        Clouds are structured arrays: data['x'], data['y'], data['z'].
        Calibration and TF are dictionaries with their exported fields.
        """
        folder = self.index_path.parent.resolve()
        path = (folder / row['file']).resolve()
        if not path.is_relative_to(folder):
            raise ValueError(f'Index path escapes recording folder: {path}')
        metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        message_type = row['type']
        if message_type == 'sensor_msgs/msg/Image':
            encoding = metadata['encoding']
            if encoding in ('rgb8', 'bgr8', 'mono8'):
                kind = 'rgb' if encoding != 'mono8' else 'grayscale'
                # The extractor already saved PNG channel order as RGB.
                with Image.open(path) as image:
                    data = np.array(image)
            else:
                kind = 'depth'
                data = np.load(path, allow_pickle=False)
        elif message_type == 'sensor_msgs/msg/PointCloud2':
            kind = 'point_cloud'
            data = np.load(path, allow_pickle=False)
        elif message_type == 'sensor_msgs/msg/CameraInfo':
            kind, data = 'camera_info', metadata
        elif message_type == 'tf2_msgs/msg/TFMessage':
            kind, data = 'transforms', metadata
        else:
            raise ValueError(f'Unsupported message type: {message_type}')
        return Measurement(
            recording=self.name, message_id=int(row['message_id']), topic=row['topic'],
            message_type=message_type, bag_timestamp_ns=int(row['bag_timestamp_ns']),
            header_timestamp_ns=int(row['header_timestamp_ns']) if row['header_timestamp_ns'] else None,
            frame_id=row['frame_id'], path=path, kind=kind, data=data, metadata=metadata)

    def measurements(self, topics: set[str] | None = None) -> Iterator[Measurement]:
        """Yield one measurement at a time in recording timestamp order.

        Independent camera streams are not synchronized or paired here.
        """
        for row in self.rows:
            if topics is None or row['topic'] in topics:
                yield self.load(row)


def discover_recordings(data_folder: Path) -> list[Recording]:
    """Find exported recordings beneath DATA_FOLDER via their index.csv files."""
    data_folder = data_folder.resolve()
    if not data_folder.is_dir():
        raise FileNotFoundError(f'Exported data folder does not exist: {data_folder}')
    indexes = sorted(data_folder.rglob('index.csv'))
    if not indexes:
        raise FileNotFoundError(f'No index.csv files found under: {data_folder}')
    return [Recording(path, path.parent.relative_to(data_folder).as_posix()) for path in indexes]


def run_recording(recording, detector, run_folder):
    rows = [row for row in recording.rows if row['topic'] == RGB_TOPIC]
    if MAX_MESSAGES is not None:
        rows = rows[:MAX_MESSAGES]
    timing = source_timing(rows)
    scenario = recording.name.split('/')[0]
    categories = CATEGORIES_BY_SCENARIO.get(scenario, OBJECT_CATEGORIES)
    if not categories or any(not isinstance(c, str) or not c.strip() for c in categories):
        raise ValueError(f'Set nonempty category prompts for {recording.name}')
    detector.categories = list(categories)
    print(f'\nTesting {recording.name}: {len(rows)} RGB frames, prompts={categories}', flush=True)

    # Warm up the model separately. Its extra calls are excluded from measured
    # inference time and video frames. Every actual RGB frame is then processed.
    warmup_start = perf_counter()
    first = recording.load(rows[0])
    for _ in range(WARMUP_FRAMES):
        detector(first)
    detector.synchronize()
    warmup_seconds = perf_counter() - warmup_start
    folder = run_folder / 'object-detection'
    folder.mkdir(parents=True, exist_ok=True)
    video = TestVideo(folder, (first.data.shape[1], first.data.shape[0]), timing['playback_fps'])
    del first
    latencies = []
    detected_classes = Counter()
    frames_with_detections = 0
    wall_start = perf_counter()
    try:
        for frame, row in enumerate(rows, 1):
            measurement = recording.load(row)
            detector.synchronize()
            start = perf_counter()
            result = detector(measurement)
            detector.synchronize()
            latency = perf_counter() - start
            latencies.append(latency)
            running_hz = len(latencies) / sum(latencies)
            frames_with_detections += bool(result.detections)
            detected_classes.update(d['class_name'] for d in result.detections)
            detector.save_result(result, folder, nest_recording=False)
            elapsed = (timing['timestamps_ns'][frame - 1] - timing['timestamps_ns'][0]) / 1e9
            video.write(detector.annotate(result), result, latency, running_hz,
                        elapsed, timing['source_fps'], frame, len(rows))
            print(f'  {frame}/{len(rows)} | last frame {latency:.1f}s | detection {running_hz:.2f} Hz | objects {len(result.detections)}', flush=True)
    finally:
        # Ensure the MP4 container is finalized even if inference raises.
        video.close()
    wall_seconds = perf_counter() - wall_start
    summary = performance_summary(latencies, wall_seconds)
    summary.update(recording=recording.name, status='complete', model=detector.model_id,
                   device=detector.device, object_categories=categories, confidence_threshold=CONFIDENCE,
                   warmup_frames=WARMUP_FRAMES, warmup_seconds=warmup_seconds,
                   configured_model_input_short_edge=MODEL_INPUT_SHORT_EDGE, configured_cpu_threads=CPU_THREADS,
                   source_clock=timing['clock'], source_fps=timing['source_fps'],
                   source_span_seconds=timing['source_span_seconds'], playback_fps=timing['playback_fps'],
                   playback_timing='constant-rate approximation using average source FPS',
                   frames_with_detections=frames_with_detections,
                   frame_detection_fraction=frames_with_detections / len(rows),
                   detected_classes=dict(detected_classes), video=str(video.path),
                   realtime_at_source_rate=(summary['detection_hz'] >= timing['source_fps'])
                   if timing['source_fps'] is not None else None)
    (folder / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"  Detection: {summary['detection_hz']:.2f} Hz | end-to-end: {summary['end_to_end_hz']:.2f} Hz", flush=True)
    print(f"  Detection: {summary['mean_detection_ms']:.2f} ms/frame | processing and export: {wall_seconds:.2f} s", flush=True)
    print(f'  Video: {video.path}', flush=True)
    return summary


def choose_recording(recordings):
    """Choose exactly one recording before loading the model."""
    if RECORDING is not None:
        matches = [r for r in recordings if RECORDING in (r.name, r.name.split('/')[0])]
        if len(matches) != 1:
            raise ValueError(f'Choose one recording. Available: {[r.name for r in recordings]}')
        return matches[0]
    print('\nChoose one test:')
    for number, recording in enumerate(recordings, 1):
        print(f'  {number}. {recording.name.split("/")[0]}')
    while True:
        answer = input('Test number (q to quit): ').strip()
        if answer.lower() == 'q':
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(recordings):
            return recordings[int(answer) - 1]
        print(f'Enter a number from 1 to {len(recordings)}, or q.')


def main():
    if MAX_MESSAGES is not None and MAX_MESSAGES < 1:
        raise ValueError('MAX_MESSAGES must be positive or None')
    if WARMUP_FRAMES < 0:
        raise ValueError('WARMUP_FRAMES must be nonnegative')
    recordings = discover_recordings(DATA_FOLDER)
    print(f'Data folder: {DATA_FOLDER.resolve()}', flush=True)
    recording = choose_recording(recordings)
    if recording is None:
        return
    print('1. Object detection only')
    print('2. Full pipeline (all stages)')
    print('3. Segmentation only (from saved detections)')
    print('4. Estimate 3D positions for Kalman (from saved masks and depth data)')
    print('5. Kalman only (from saved 3D measurements)')
    print('Timing is recorded for every test.')
    while True:
        test_mode = input('Test type (1-5, q to quit): ').strip().lower()
        if test_mode == 'q':
            return
        if test_mode in ('1', '2', '3', '4', '5'):
            break
    if test_mode != '1':
        from object_pose_estimator import offline_runner
        if test_mode == '2':
            offline_runner.full_pipeline_test(recording, sys.modules[__name__])
            return
        source = offline_runner.select_stage_input(recording, test_mode)
        if source is None:
            return
        if test_mode == '3':
            offline_runner.segment_saved(recording, source)
        elif test_mode == '4':
            offline_runner.prepare_cache(recording=recording, segments_folder=source)
        else:
            offline_runner.replay_filter(source, recording=recording)
        return
    selected = [recording]
    model_start = perf_counter()
    detector = ObjectDetector(GDINO_MODEL, CONFIDENCE, DEVICE, OBJECT_CATEGORIES)
    print(f'Model device: {detector.device}', flush=True)
    detector.configure_cpu(MODEL_INPUT_SHORT_EDGE, CPU_THREADS)
    model_load_seconds = perf_counter() - model_start
    run_folder = OUTPUT_FOLDER / recording.name / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8] + '_detection')
    run_folder.mkdir(parents=True, exist_ok=False)
    print(f'Test output: {run_folder}', flush=True)
    summaries = []
    for recording in selected:
        summaries.append(run_recording(recording, detector, run_folder))
        # Save after every completed recording to preserve progress.
        (run_folder / 'summary.json').write_text(json.dumps(
            dict(model_load_seconds=model_load_seconds, recordings=summaries), indent=2), encoding='utf-8')
        with (run_folder / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
            columns = ['recording', 'frames', 'source_fps', 'detection_hz', 'end_to_end_hz',
                       'mean_detection_ms', 'median_detection_ms', 'frame_detection_fraction', 'video']
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(summaries)
    print('Finished. Detection fraction is not accuracy: no ground-truth evaluation was performed.', flush=True)
    if OPEN_VIDEO_WHEN_DONE and os.name == 'nt' and summaries:
        try:
            os.startfile(summaries[0]['video'])
        except OSError as error:
            print(f"Could not open player automatically: {error}. Open the video path above.")


if __name__ == '__main__':
    main()
