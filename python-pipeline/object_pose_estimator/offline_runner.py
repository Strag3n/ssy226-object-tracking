"""Run saved detections -> original SAM2/3D measurements -> original Kalman.

Run as object_pose_estimator.offline_runner from the pipeline folder.
Menu 1 prepares/caches 3D measurements once.
Menu 2 replays only the original Kalman filter from that cache (no models).
The offline runner imports the unmodified pose-estimation modules.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path, PureWindowsPath
from bisect import bisect_right
from datetime import datetime
from uuid import uuid4
from time import perf_counter
import json
import csv
import hashlib
import numpy as np
from PIL import Image, ImageDraw

from .constants import (
    DEPTH_MODEL,
    ESTIMATION_RATE_HZ,
    FULL_OCCLUSION_THRESHOLD,
    MAX_OCCLUDED_AGE_S,
    MEASUREMENT_NOISE_POSITION,
    MIN_MASK_AREA_PX,
    MIN_SAM_SCORE,
    OCCLUSION_R_SCALE,
    PARTIAL_OCCLUSION_THRESHOLD,
    PROCESS_NOISE_POSITION,
    PROCESS_NOISE_VELOCITY,
    SAM2_MODEL,
    SEGMENTATION_HEIGHT,
    SEGMENTATION_WIDTH,
    VELOCITY_ZERO_THRESH,
)
from .kalman_tracker import KalmanTracker
from .occlusion_state_machine import OcclusionStateMachine, TrackingState
from .pose_helpers import (
    _correct_centroid_partial_occ,
    _select_z,
    _z_from_depth_map,
    _z_from_pointcloud,
)

# ==================== EDIT SETTINGS HERE ====================
PIPELINE_FOLDER = Path(__file__).resolve().parents[1]
DATA_FOLDER = PIPELINE_FOLDER.parent / 'extracted-data'
DETECTION_FOLDER = PIPELINE_FOLDER.parent / 'results'
# Original constants.py supplies dt=1/5 s and filter noises 0.5, 20, 0.005.
# For baseline, leave those constants unchanged.
DEVICE = 'cpu'
MAX_STEPS = None  # e.g. 5 for initial SAM2/depth setup check; None for all.
REQUIRE_BASELINE = False  # True rejects reduced/unknown detection resolution.
# ============================================================

def choose(items, label):
    if not items:
        raise FileNotFoundError(f'No {label} found')
    print(f'\nChoose {label}:')
    for number, (name, _) in enumerate(items, 1):
        print(f'  {number}. {name}')
    while True:
        answer = input('Number (q to quit): ').strip()
        if answer.lower() == 'q':
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(items):
            return items[int(answer)-1][1]
        print('Enter a number shown above.')


def load_saved(folder):
    records = []
    for path in folder.glob('*.json'):
        record = json.loads(path.read_text(encoding='utf-8'))
        if 'detections' in record and 'source_image' in record:
            record['_json_path'] = str(path)
            records.append(record)
    records.sort(key=lambda row: (row['bag_timestamp_ns'], row['message_id']))
    if not records:
        raise ValueError('This folder has no saved detection frames')
    if len({row['recording'] for row in records}) != 1:
        raise ValueError('Select exactly one recording folder')
    return records


def timer_samples(records):
    """Replay the original 5 Hz timer using the latest available saved RGB.

    Saved bag timestamps, not inference wall times, define offline timing.
    Never choose a detection from the future. Keep original Kalman dt fixed.
    """
    times = [row['bag_timestamp_ns'] for row in records]
    step_ns = round(1e9 / ESTIMATION_RATE_HZ)
    selected = []
    for tick in range(times[0], times[-1] + 1, step_ns):
        row = records[bisect_right(times, tick) - 1]
        selected.append((tick, row))
    return selected[:MAX_STEPS] if MAX_STEPS is not None else selected


def synchronize_device(device):
    import torch
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    elif str(device).startswith('mps'):
        torch.mps.synchronize()


class PositionPreparation:
    """Offline positional portion of the original pose estimator.

    No rotation/gripper/TF tracking is claimed here. Saved measurements are
    in the recorded camera frame, which may move when the robot moves.
    """
    def __init__(self, device=DEVICE, *, use_saved_masks=False):
        self.device = device
        self.segmenter = None if use_saved_masks else make_segmenter(device)
        from .depth_estimator import DepthEstimator

        self.depth = DepthEstimator(device=device)
        self.state_machine = OcclusionStateMachine(PARTIAL_OCCLUSION_THRESHOLD, FULL_OCCLUSION_THRESHOLD)
        self._ref_mask_area = 0
        self._ref_z = self._last_z = None
        self._clean_bbox = None
        self._ref_box_W = self._ref_box_H = 0.0
        self.scale_state = {'ratio': 1.0, 'calibrated': False}
        self.occluded_since = None


    def measure(self, rgb, detections, xyz, K, timestamp_ns, *, saved_mask=None):
        synchronize_device(self.device)
        stage_start = perf_counter()
        mask, bbox, sam_score = (self.segmenter.segment(rgb, detections)
                                 if saved_mask is None else saved_mask)
        synchronize_device(self.device)
        segmentation_ms = (perf_counter() - stage_start) * 1000
        remaining_start = perf_counter()
        depth_ms = 0.0
        depth_executed = False
        self.last_depth_map = None
        succeeded = mask is not None
        occlusion_ratio = 0.0
        if succeeded:
            current_area = int(mask.sum())
            if current_area > self._ref_mask_area:
                self._ref_mask_area = current_area
                self._ref_z = self._last_z
            if self._ref_mask_area > 0:
                if self._ref_z and self._last_z:
                    z_ratio = float(np.clip(self._last_z / self._ref_z, 0.8, 1.2))
                    expected_area = self._ref_mask_area / (z_ratio ** 2)
                else:
                    expected_area = self._ref_mask_area
                occlusion_ratio = max(0.0, 1.0 - current_area / expected_area)
        # No /joint_states was recorded; false matches original missing-state behavior.
        state = self.state_machine.update(occlusion_ratio, succeeded, False)
        position = None
        reset = False
        z_pc = z_da = z_selected = None
        depth_source = None
        if state in (TrackingState.TRACKING, TrackingState.PARTIALLY_OCCLUDED):
            self.occluded_since = None
            z_pc = _z_from_pointcloud(xyz, mask) if xyz is not None else None
            synchronize_device(self.device)
            depth_start = perf_counter()
            depth_map = self.depth.estimate(rgb)
            self.last_depth_map = depth_map
            synchronize_device(self.device)
            depth_ms = (perf_counter() - depth_start) * 1000
            depth_executed = True
            z_da = _z_from_depth_map(mask, depth_map)
            z_selected, depth_source = _select_z(z_pc, z_da, self.scale_state)
            if z_selected is not None and z_selected > 0:
                self._last_z = z_selected
                height, width = mask.shape
                if (self._clean_bbox is not None and self._ref_box_W > 0 and occlusion_ratio > 0.05):
                    u, v = _correct_centroid_partial_occ(
                        *bbox, self._clean_bbox, self._ref_box_W, self._ref_box_H,
                        img_W=width, img_H=height)
                else:
                    x1, y1, x2, y2 = bbox
                    u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                fx, fy, cx_cam, cy_cam = K[0,0], K[1,1], K[0,2], K[1,2]
                position = np.array([(u-cx_cam)*z_selected/fx, (v-cy_cam)*z_selected/fy, z_selected])
                if bbox is not None and occlusion_ratio <= 0.05:
                    self._clean_bbox = tuple(bbox)
                    self._ref_box_W = max(self._ref_box_W, float(bbox[2]-bbox[0]))
                    self._ref_box_H = max(self._ref_box_H, float(bbox[3]-bbox[1]))
        else:
            now = timestamp_ns / 1e9  # Replay source time instead of processing wall time.
            if self.occluded_since is None:
                self.occluded_since = now
            if now - self.occluded_since > MAX_OCCLUDED_AGE_S:
                reset = True
                self._ref_mask_area = 0
                self._ref_z = self._last_z = None
                self.occluded_since = None
        if position is not None and not np.isfinite(position).all():
            raise ValueError('Original position calculation returned NaN/Inf; refusing to cache invalid measurements')
        self.last_timings = dict(segmentation_ms=segmentation_ms, depth_ms=depth_ms,
                                 position_occlusion_ms=max(0.0, (perf_counter()-remaining_start)*1000-depth_ms),
                                 depth_executed=depth_executed)
        return dict(position=position.tolist() if position is not None else None,
                    occlusion_ratio=float(occlusion_ratio), state=state.name, reset=reset,
                    sam_score=sam_score, z_pointcloud=z_pc, z_depth_anything=z_da,
                    z_selected=z_selected, depth_source=depth_source), mask


def latest_row(rows, timestamp):
    times = [int(row['bag_timestamp_ns']) for row in rows]
    index = bisect_right(times, timestamp)-1
    return rows[index] if index >= 0 else None


def detection_provenance(folder):
    path = folder / 'metrics.json'
    if not path.exists():
        path = folder.parent / 'timing' / 'timing_summary.json'
    if not path.exists():
        path = folder.parents[1] / 'timing_summary.json'
    metrics = json.loads(path.read_text()) if path.exists() else {}
    key = 'configured_model_input_short_edge'
    status = ('unknown' if key not in metrics else
              'original_preprocessing' if metrics[key] is None else 'modified_preprocessing')
    return dict(resolution_status=status, configured_model_input_short_edge=metrics.get(key))


def make_segmenter(device):
    from .observation_models import Segmenter

    return Segmenter(SAM2_MODEL, device, SEGMENTATION_WIDTH,
                     SEGMENTATION_HEIGHT, MIN_SAM_SCORE, MIN_MASK_AREA_PX)


def segment_saved(recording, folder):
    saved = load_saved(folder)
    if saved[0]['recording'] != recording.name:
        raise ValueError('Saved detections belong to a different recording.')
    frames = {(int(row['message_id']), int(row['bag_timestamp_ns'])): row for row in recording.rows}
    load_start = perf_counter()
    segmenter = make_segmenter(DEVICE)
    model_load_seconds = perf_counter() - load_start
    output = new_run(DETECTION_FOLDER, 'segmentation', recording.name)
    masks = output / 'masks'
    masks.mkdir()
    samples = timer_samples(saved)
    masks_written = 0
    durations = []
    wall_start = perf_counter()
    with (output / 'segments.jsonl').open('w', encoding='utf-8') as handle:
        for tick, frame in samples:
            key = frame['message_id'], frame['bag_timestamp_ns']
            image = recording.load(frames[key])
            synchronize_device(DEVICE)
            start = perf_counter()
            mask, bbox, score = segmenter.segment(image.data, frame['detections'])
            synchronize_device(DEVICE)
            duration = (perf_counter() - start) * 1000
            durations.append(duration)
            mask_file = None
            if mask is not None:
                mask_file = f'masks/{tick}.npy'
                np.save(output / mask_file, mask, allow_pickle=False)
                masks_written += 1
            handle.write(json.dumps(dict(
                timestamp_ns=tick, message_id=key[0], bag_timestamp_ns=key[1],
                mask=mask_file, bbox=np.asarray(bbox).tolist() if bbox is not None else None,
                sam_score=score, segmentation_ms=duration), allow_nan=False) + '\n')
    wall_seconds = perf_counter() - wall_start
    (output / 'manifest.json').write_text(json.dumps(dict(
        status='complete', recording=recording.name, source_detection_folder=str(folder.resolve()),
        dt=1/ESTIMATION_RATE_HZ, sam_model=SAM2_MODEL, frames=len(samples),
        frames_with_masks=masks_written, model_load_seconds=model_load_seconds,
        segmentation_ms=timing_stats(durations), end_to_end_seconds=wall_seconds), indent=2), encoding='utf-8')
    print(f'Segmentation: {np.mean(durations):.2f} ms/frame | processing and export: {wall_seconds:.2f} s')
    print(f'Segmentation output: {output}')
    return output


def new_run(root, stage, recording):
    run = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
    run_folder = root / recording / (run + '_' + stage)
    output = run_folder if stage == 'full-pipeline' else run_folder / stage
    output.mkdir(parents=True, exist_ok=False)
    return output


def timing_stats(durations):
    if not durations:
        return None
    return dict(mean_ms=float(np.mean(durations)), median_ms=float(np.median(durations)),
                p95_ms=float(np.percentile(durations, 95)))


def prepare_cache(folder=None, *, recording=None, segments_folder=None):
    segments = None
    if segments_folder is not None:
        manifest = json.loads((segments_folder / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['status'] != 'complete' or not np.isclose(manifest['dt'], 1 / ESTIMATION_RATE_HZ):
            raise ValueError('Choose a completed segmentation run at the current sampling rate.')
        folder = Path(manifest['source_detection_folder'])
        segments = {row['timestamp_ns']: row for row in (
            json.loads(line) for line in (segments_folder / 'segments.jsonl').read_text(
                encoding='utf-8').splitlines())}
    saved = load_saved(folder)
    detection_setup = detection_provenance(folder)
    if REQUIRE_BASELINE and detection_setup['resolution_status'] != 'original_preprocessing':
        raise ValueError('Strict baseline mode: rerun detection with MODEL_INPUT_SHORT_EDGE=None, or set REQUIRE_BASELINE=False for experiments.')
    print(f"Detection resolution: {detection_setup['resolution_status']}", flush=True)
    if detection_setup['resolution_status'] != 'original_preprocessing':
        print('EXPERIMENT: reusing these detections; do not label this an original-resolution baseline.', flush=True)
    category = ('all saved categories' if segments is not None else
                choose([(name, name) for name in saved[0]['object_categories']], 'object category'))
    if category is None:
        return
    name = saved[0]['recording']
    if recording is not None and recording.name != name:
        raise ValueError('Saved detections belong to a different recording.')
    if recording is None:
        from start import discover_recordings

        recording = next((r for r in discover_recordings(DATA_FOLDER) if r.name == name), None)
    if recording is None:
        raise FileNotFoundError(f'Original extracted recording not found: {name}')
    cloud_rows = [r for r in recording.rows if r['type']=='sensor_msgs/msg/PointCloud2']
    info_rows = [r for r in recording.rows if r['type']=='sensor_msgs/msg/CameraInfo']
    if not cloud_rows or not info_rows:
        raise ValueError('Original pipeline requires point clouds and RGB CameraInfo')
    load_start = perf_counter()
    model = PositionPreparation(use_saved_masks=segments is not None)
    model_load_seconds = perf_counter() - load_start
    output = new_run(DETECTION_FOLDER, '3d-measurements', name)
    manifest = dict(status='preparing', source_detection_folder=str(folder.resolve()), recording=name,
                    detection_preprocessing=detection_setup,
                    category=category, sam_model=SAM2_MODEL, depth_model=DEPTH_MODEL,
                    source_segmentation_folder=str(segments_folder) if segments_folder else None,
                    dt=1/ESTIMATION_RATE_HZ, sampling='5Hz timer, latest available saved RGB/cloud; bag-time causal',
                    gripper_available=False, rotation_included=False, wall_time_equivalence=False,
                    cloud_registration_verified=False, cloud_alignment="same pixel indexing as original; registration assumed",
                    frame_id=None, parameters=dict(process_noise_pos=PROCESS_NOISE_POSITION,
                    process_noise_vel=PROCESS_NOISE_VELOCITY, measurement_noise=MEASUREMENT_NOISE_POSITION))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    K = None
    camera_frame = None
    written = 0
    step_durations = []
    depth_durations = []
    samples = timer_samples(saved)
    if segments is not None:
        saved_by_key = {(row['message_id'], row['bag_timestamp_ns']): row for row in saved}
        samples = [(tick, saved_by_key[row['message_id'], row['bag_timestamp_ns']])
                   for tick, row in sorted(segments.items())]
    wall_start = perf_counter()
    with (output/'measurements.jsonl').open('w',encoding='utf-8') as handle:
        for number, (tick, frame) in enumerate(samples, 1):
            frame_start = perf_counter()
            cloud_row = latest_row(cloud_rows, tick)
            info_row = latest_row(info_rows, tick)
            if cloud_row is None or (K is None and info_row is None):
                continue  # Same as the original node waiting for required inputs.
            if K is None:
                info = recording.load(info_row)
                K = np.asarray(info.data['K'], dtype=float).reshape(3,3)
                camera_frame = info.data['frame_id']
                if K[0,0] <= 0 or K[1,1] <= 0:
                    raise ValueError('Invalid camera intrinsics')
            rgb_path = next((DATA_FOLDER / name / r['file'] for r in recording.rows
                             if int(r['message_id'])==frame['message_id']), None)
            if rgb_path is None:
                raise FileNotFoundError('Saved image cannot be matched to extracted index')
            with Image.open(rgb_path) as image:
                rgb = np.asarray(image.convert('RGB'))
            cloud = recording.load(cloud_row)
            frame_mismatch = cloud.frame_id != frame['frame_id'] or camera_frame != frame['frame_id']
            if frame_mismatch and written == 0:
                print('WARNING: RGB/cloud frame names differ. Retaining the original pixel-index alignment assumption; 3D registration is NOT verified.', flush=True)
            xyz = np.stack([cloud.data[f] for f in ('x','y','z')],axis=-1).astype(np.float32)
            if xyz.shape[:2] != rgb.shape[:2]:
                raise ValueError('Organized cloud does not match RGB dimensions')
            detections = [d for d in frame['detections'] if d['class_name']==category]
            start = perf_counter()
            options = {}
            if segments is not None:
                segment = segments[tick]
                if (segment['message_id'], segment['bag_timestamp_ns']) != (
                        frame['message_id'], frame['bag_timestamp_ns']):
                    raise ValueError(f'Saved mask belongs to a different RGB frame at {tick}.')
                mask = (np.load(segments_folder / segment['mask'], allow_pickle=False)
                        if segment['mask'] else None)
                options['saved_mask'] = mask, segment['bbox'], segment['sam_score']
            result, mask = model.measure(rgb, detections, xyz, K, tick, **options)
            result.update(timestamp_ns=tick, source_message_id=frame['message_id'],
                          source_bag_timestamp_ns=frame['bag_timestamp_ns'], image=str(rgb_path.resolve()),
                          frame_id=frame['frame_id'], cloud_frame_id=cloud.frame_id, calibration_frame_id=camera_frame,
                          coordinate_frame_mismatch=frame_mismatch, registration_verified=False,
                          K=K.tolist(), cloud_age_seconds=(tick-cloud.bag_timestamp_ns)/1e9,
                          detection_json=frame['_json_path'],
                          detection_sha256=hashlib.sha256(Path(frame['_json_path']).read_bytes()).hexdigest())
            if mask is not None:
                mask_path = output/f'mask_{number:05d}.npy'
                np.save(mask_path, mask, allow_pickle=False)
                result['mask']=mask_path.name
            timings = model.last_timings
            frame_ms = (perf_counter() - frame_start) * 1000
            step_durations.append(frame_ms)
            if timings['depth_executed']:
                depth_durations.append(timings['depth_ms'])
            result.update(timings, frame_before_write_ms=frame_ms)
            handle.write(json.dumps(result,allow_nan=False)+'\n')
            handle.flush()
            written += 1
            print(f'3D cache {number}: {result["state"]}, position={result["position"]}, {perf_counter()-start:.1f}s',flush=True)
    if not written:
        raise ValueError('No frames had the required causal inputs; cache remains incomplete')
    wall_seconds = perf_counter() - wall_start
    manifest.update(status='complete', frames=written, frame_id=camera_frame,
                    model_load_seconds=model_load_seconds, frame_before_write_ms=timing_stats(step_durations),
                    depth_ms=timing_stats(depth_durations), depth_executed_frames=len(depth_durations),
                    end_to_end_seconds=wall_seconds)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'3D positions: {np.mean(step_durations):.2f} ms/frame | processing and export: {wall_seconds:.2f} s')
    print(f'3D measurements cached: {output}\nChoose Kalman only in start.py to replay them.')
    return output


def filter_step(tracker, row):
    """Original positional initialize/predict/update/reset decisions."""
    if row['reset']:
        tracker.reset()
        return None, 'reset'
    position = np.array(row['position'],dtype=float) if row['position'] is not None else None
    if row['state'] in ('TRACKING','PARTIALLY_OCCLUDED'):
        if position is not None:
            if not tracker.initialized:
                tracker.initialize(position)
                return position.copy(), 'initialize'
            tracker.predict()
            return tracker.update(position, row['occlusion_ratio'], OCCLUSION_R_SCALE), 'update'
        if tracker.initialized:
            return tracker.predict(), 'predict_no_position'
    elif row['state']=='FULLY_OCCLUDED' and tracker.initialized:
        if np.linalg.norm(tracker.get_velocity()) < VELOCITY_ZERO_THRESH:
            tracker.zero_velocity()
        return tracker.predict(), 'predict_occluded'
    return None, 'waiting'


def replay_filter(folder, *, recording=None):
    import cv2
    manifest_path = folder / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    else:
        summary_path = folder / 'timing_summary.json'
        if not summary_path.is_file():
            summary_path = folder.parent / 'timing' / 'timing_summary.json'
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        manifest = dict(status='complete', recording=summary['recording'], dt=summary['kalman_dt'])
    if manifest['status'] != 'complete':
        raise ValueError('Choose a completed cache')
    if not np.isclose(manifest['dt'],1/ESTIMATION_RATE_HZ):
        raise ValueError('Cache sampling rate differs from current filter dt')
    rows = [json.loads(line) for line in (folder/'measurements.jsonl').read_text().splitlines()]
    if not rows:
        raise ValueError('Empty measurement cache')
    output = new_run(DETECTION_FOLDER, 'kalman-only', manifest['recording'])
    index = ({Path(row['file']).name: recording.index_path.parent / row['file']
              for row in recording.rows} if recording is not None else {})
    tracker = KalmanTracker(1/ESTIMATION_RATE_HZ, PROCESS_NOISE_POSITION,
                            PROCESS_NOISE_VELOCITY, MEASUREMENT_NOISE_POSITION)
    writer = None
    elapsed_filter = 0.0
    wall_start=perf_counter()
    fields=['timestamp_ns','state','mode','occlusion_ratio','measured_x','measured_y','measured_z',
            'filtered_x','filtered_y','filtered_z','vx','vy','vz','filter_ms']
    try:
        with (output/'kalman.csv').open('w',newline='') as handle:
            csv_writer=csv.DictWriter(handle,fieldnames=fields)
            csv_writer.writeheader()
            for i,row in enumerate(rows,1):
                start=perf_counter()
                filtered,mode=filter_step(tracker,row)
                latency=perf_counter()-start
                elapsed_filter+=latency
                record=dict(timestamp_ns=row['timestamp_ns'],state=row['state'],mode=mode,
                            occlusion_ratio=row['occlusion_ratio'],filter_ms=1000*latency)
                for prefix,point in [('measured',row['position']),('filtered',filtered)]:
                    for j,axis in enumerate('xyz'):
                        record[f'{prefix}_{axis}']=float(point[j]) if point is not None else ''
                record.update(dict(zip(('vx','vy','vz'),tracker.get_velocity())))
                csv_writer.writerow(record)
                image_path = index.get(PureWindowsPath(row['image']).name, Path(row['image']))
                with Image.open(image_path) as source:
                    image=source.convert('RGB')
                draw=ImageDraw.Draw(image)
                K=np.array(row['K'])
                for point,color,label in [(row['position'],'red','Measured 3D'),(filtered,'lime','Kalman 3D')]:
                    if point is not None and point[2]>0:
                        u=K[0,0]*point[0]/point[2]+K[0,2]
                        v=K[1,1]*point[1]/point[2]+K[1,2]
                        draw.ellipse((u-5,v-5,u+5,v+5),fill=color)
                text=f'{i}/{len(rows)} | {row["state"]} | {mode} | Kalman {latency*1000:.3f}ms'
                draw.rectangle(draw.textbbox((5,5),text),fill='black')
                draw.text((5,5),text,fill='white')
                draw.text((5,22),'Red: measured 3D | Green: filtered 3D | playback 5 FPS',fill='white',stroke_fill='black',stroke_width=1)
                if writer is None:
                    writer=cv2.VideoWriter(str(output/'kalman.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),ESTIMATION_RATE_HZ,image.size)
                    if not writer.isOpened():
                        raise RuntimeError('MP4 encoder could not open')
                writer.write(np.ascontiguousarray(np.asarray(image)[...,::-1]))
    finally:
        if writer is not None:
            writer.release()
    summary=dict(recording=manifest['recording'], cache=str(folder.resolve()),frames=len(rows),filter_only_hz=len(rows)/elapsed_filter,
                 filter_seconds=elapsed_filter,end_to_end_seconds=perf_counter()-wall_start,
                 dt=tracker.dt,process_noise_pos=PROCESS_NOISE_POSITION,
                 process_noise_vel=PROCESS_NOISE_VELOCITY,measurement_noise=MEASUREMENT_NOISE_POSITION,
                 actual_model_calls=0,accuracy_evaluated=False,
                 detection_preprocessing=manifest.get('detection_preprocessing', {'resolution_status':'unknown'}))
    (output/'metrics.json').write_text(json.dumps(summary,indent=2))
    print(f'Kalman: {elapsed_filter * 1000 / len(rows):.3f} ms/step | '
          f'processing and export: {summary["end_to_end_seconds"]:.2f} s\nVideo: {output/"kalman.mp4"}')
    return output


def select_stage_input(recording, mode):
    root = DETECTION_FOLDER
    candidates = set()
    if mode == '3':
        for path in root.rglob('*.json'):
            try:
                info = json.loads(path.read_text(encoding='utf-8'))
            except (ValueError, OSError):
                continue
            if info.get('recording') == recording.name and 'source_image' in info and 'detections' in info:
                candidates.add(path.parent)
    else:
        for path in root.rglob('manifest.json'):
            info = json.loads(path.read_text(encoding='utf-8'))
            required = 'segments.jsonl' if mode == '4' else 'measurements.jsonl'
            if (info.get('status') == 'complete' and info.get('recording') == recording.name
                    and (path.parent / required).is_file()):
                candidates.add(path.parent)
        if mode == '5':
            for path in root.rglob('timing_summary.json'):
                info = json.loads(path.read_text(encoding='utf-8'))
                if info.get('recording') == recording.name:
                    for folder in (path.parent, path.parent.parent / 'kalman'):
                        if (folder / 'measurements.jsonl').is_file():
                            candidates.add(folder)
    if not candidates:
        prerequisite = {'3': '1 (detection) or 2 (full pipeline)',
                        '4': '3 (segmentation)', '5': '4 (3D measurements) or 2 (full pipeline)'}[mode]
        print(f'No saved input for {recording.name}. Run option {prerequisite} for this recording first.')
        return None
    return choose([(path.relative_to(DATA_FOLDER.parent.parent).as_posix(), path)
                   for path in sorted(candidates)], 'saved input run')


def full_pipeline_test(recording, settings):
    """Timed single-recording offline test. No realtime/rotation claim."""
    from .observation_models import ObjectDetector
    from dataclasses import replace
    from video import TestVideo
    raw = [dict(r, bag_timestamp_ns=int(r['bag_timestamp_ns'])) for r in recording.rows
           if r['topic']==settings.RGB_TOPIC]
    if settings.MAX_MESSAGES is not None:
        raw = raw[:settings.MAX_MESSAGES]
    if not raw:
        raise ValueError('No selected RGB frames')
    samples = timer_samples(raw)
    clouds = [r for r in recording.rows if r['type']=='sensor_msgs/msg/PointCloud2']
    infos = [r for r in recording.rows if r['type']=='sensor_msgs/msg/CameraInfo']
    if not clouds or not infos:
        raise ValueError('Point clouds and camera calibration are required')
    categories = settings.CATEGORIES_BY_SCENARIO.get(recording.name.split('/')[0],settings.OBJECT_CATEGORIES)
    load_start=perf_counter()
    detector=ObjectDetector(settings.GDINO_MODEL,settings.CONFIDENCE,settings.DEVICE,categories)
    detector.configure_cpu(settings.MODEL_INPUT_SHORT_EDGE,settings.CPU_THREADS)
    detection_load=perf_counter()-load_start
    load_start=perf_counter()
    preparation=PositionPreparation(detector.device)
    preparation_load=perf_counter()-load_start
    tracker=KalmanTracker(1/ESTIMATION_RATE_HZ,PROCESS_NOISE_POSITION,PROCESS_NOISE_VELOCITY,MEASUREMENT_NOISE_POSITION)
    output = new_run(settings.OUTPUT_FOLDER, 'full-pipeline', recording.name)
    for name in ('object-detection', 'segmentation', 'depth-model', '3d-occlusion', 'kalman', 'timing', 'video', 'plots'):
        (output / name).mkdir(exist_ok=True)
    video=None
    timings=[]
    modes={}
    K=None
    clock_start=perf_counter()
    print(f'Full pipeline: {recording.name}, {len(samples)} scheduled steps at {ESTIMATION_RATE_HZ} Hz (offline)',flush=True)
    fields=['step','timestamp_ns','source_message_id','load_ms','detection_ms','segmentation_ms',
            'depth_ms','depth_executed','position_occlusion_ms','kalman_ms','kalman_mode',
            'objects','state','compute_ms','export_ms','end_to_end_ms']
    try:
        with (output/'timing'/'stage_timings.csv').open('w',newline='') as handle, (output/'kalman'/'measurements.jsonl').open('w') as measured:
            csvwriter=csv.DictWriter(handle,fieldnames=fields)
            csvwriter.writeheader()
            for number,(tick,row) in enumerate(samples,1):
                frame_start=perf_counter()
                cloud_row=latest_row(clouds,tick)
                info_row=latest_row(infos,tick)
                if cloud_row is None or (K is None and info_row is None):
                    continue
                image=recording.load(row)
                cloud=recording.load(cloud_row)
                if K is None:
                    info=recording.load(info_row)
                    K=np.array(info.data['K'],dtype=float).reshape(3,3)
                xyz=np.stack([cloud.data[f] for f in ('x','y','z')],axis=-1).astype(np.float32)
                if xyz.shape[:2]!=image.data.shape[:2]:
                    raise ValueError('Cloud dimensions differ from image')
                if not timings and cloud.frame_id!=image.frame_id:
                    print('WARNING: RGB/cloud frame names differ; retaining original pixel-index alignment assumption.',flush=True)
                load_ms=(perf_counter()-frame_start)*1000
                detector.synchronize()
                start=perf_counter()
                detected=detector(image)
                detector.synchronize()
                detection_ms=(perf_counter()-start)*1000
                measurement,mask=preparation.measure(image.data,detected.detections,xyz,K,tick)
                stage=dict(preparation.last_timings)
                start=perf_counter()
                filtered,mode=filter_step(tracker,measurement)
                kalman_ms=(perf_counter()-start)*1000
                modes[mode]=modes.get(mode,0)+1
                compute_ms=detection_ms+stage['segmentation_ms']+stage['depth_ms']+stage['position_occlusion_ms']+kalman_ms
                start=perf_counter()
                detector.save_result(detected, output / 'object-detection', nest_recording=False)
                stem = f'{tick}_{image.message_id}'
                if mask is not None:
                    np.save(output / 'segmentation' / (stem + '.npy'), mask, allow_pickle=False)
                if preparation.last_depth_map is not None:
                    np.save(output / 'depth-model' / (stem + '.npy'), preparation.last_depth_map, allow_pickle=False)
                (output / '3d-occlusion' / (stem + '.json')).write_text(
                    json.dumps(dict(measurement, timestamp_ns=tick, source_message_id=image.message_id), allow_nan=False))
                measurement.update(timestamp_ns=tick,image=str(image.path),K=K.tolist(),
                                   filtered_position=filtered.tolist() if filtered is not None else None,
                                   detections=list(detected.detections),kalman_mode=mode)
                measured.write(json.dumps(measurement,allow_nan=False)+'\n')
                measured.flush()
                from pipeline_visualization import pipeline_overlay
                view = pipeline_overlay(detector.annotate(detected), mask, measurement,
                                        dict(stage, detection_ms=detection_ms, kalman_ms=kalman_ms))
                if video is None:
                    video=TestVideo(output / 'video',view.size,ESTIMATION_RATE_HZ, filename='pipeline.mp4')
                latency=max(detection_ms/1000,1e-12)
                running_detection_hz=(len(timings)+1)*1000/(sum(t["detection_ms"] for t in timings)+detection_ms)
                video.write(view,detected,latency,running_detection_hz,(tick-samples[0][0])/1e9,
                            ESTIMATION_RATE_HZ,number,len(samples))
                export_ms=(perf_counter()-start)*1000
                item=dict(step=number,timestamp_ns=tick,source_message_id=image.message_id,
                          load_ms=load_ms,detection_ms=detection_ms,**stage,kalman_ms=kalman_ms,
                          kalman_mode=mode,objects=len(detected.detections),state=measurement['state'],
                          compute_ms=compute_ms,export_ms=export_ms,end_to_end_ms=(perf_counter()-frame_start)*1000)
                csvwriter.writerow(item)
                handle.flush()
                timings.append(item)
                print(f"{number}/{len(samples)} | DINO {detection_ms/1000:.2f}s | SAM {stage['segmentation_ms']/1000:.2f}s | depth {stage['depth_ms']/1000:.2f}s | Kalman {kalman_ms:.3f}ms ({mode})",flush=True)
    finally:
        if video is not None:
            video.close()
    if not timings:
        raise ValueError('No steps had required input data')
    columns=['load_ms','detection_ms','segmentation_ms','depth_ms','position_occlusion_ms','kalman_ms','compute_ms','export_ms','end_to_end_ms']
    def statistics(items):
        return {key:dict(mean_ms=float(np.mean([r[key] for r in items])),
                         median_ms=float(np.median([r[key] for r in items])),
                         p95_ms=float(np.percentile([r[key] for r in items],95))) for key in columns} if items else {}
    summary=dict(recording=recording.name,frames=len(timings),device=detector.device,
                 configured_model_input_short_edge=settings.MODEL_INPUT_SHORT_EDGE,
                 detection_model=settings.GDINO_MODEL,sam_model=SAM2_MODEL,depth_model=DEPTH_MODEL,
                 confidence=settings.CONFIDENCE,object_categories=categories,kalman_modes=modes,
                 scheduled_rate_hz=ESTIMATION_RATE_HZ,kalman_dt=tracker.dt,
                 first_frame=timings[0],all_frames=statistics(timings),
                 excluding_first_frame=statistics(timings[1:]),
                 detection_model_load_seconds=detection_load,sam_and_depth_model_load_seconds=preparation_load,
                 cold_start_policy='No hidden warmup; first processed frame is reported separately and excluded from steady statistics',
                 depth_executed_frames=sum(r['depth_executed'] for r in timings),
                 rotation_filter_included=False,cloud_registration_verified=False,
                 offline_wall_seconds=perf_counter()-clock_start,accuracy_evaluated=False,
                 video=str(output/'video'/'pipeline.mp4'))
    (output/'timing'/'timing_summary.json').write_text(json.dumps(summary,indent=2))
    print(f'Full pipeline output: {output}')
    for stage,result in summary['all_frames'].items():
        print(f"{stage}: mean {result['mean_ms']:.3f} ms",flush=True)
    print(f"Processing and export: {summary['offline_wall_seconds']:.2f} s", flush=True)
    print(f'Timing results: {output}',flush=True)
    return output


