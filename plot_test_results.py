"""Plot saved test results: reference points, detections, Kalman estimates and processing times."""

import argparse
import csv
import json
from pathlib import Path, PureWindowsPath
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# VS Code: open this file and choose Run Python File.
BASELINE = Path(__file__).resolve().parent
RESULTS = BASELINE / 'results'
CATEGORY = 'apple'


def find_run(folder):
    """Find the enclosing run without inspecting sibling test runs."""
    for parent in (folder, *folder.parents):
        if any((parent / name).is_file() for name in
               ('timing/stage_timings.csv', 'stage_timings.csv', 'summary.csv', 'summary.json')):
            return parent
        if parent == RESULTS or parent == BASELINE:
            break
    return folder


def choose_saved_run():
    """Select a run folder and infer its recording and matching saved inputs."""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()
    try:
        while True:
            selected = filedialog.askdirectory(
                parent=root, title='Choose a test run folder (or its recording subfolder)',
                initialdir=str(RESULTS if RESULTS.is_dir() else BASELINE))
            if not selected:
                return None
            folder = Path(selected)
            groups = {}
            category_counts = {}
            for path in folder.rglob('*.json'):
                try:
                    row = json.loads(path.read_text(encoding='utf-8'))
                except (OSError, ValueError):
                    continue
                if 'detections' in row and 'source_image' in row:
                    groups.setdefault((path.parent, row['recording']), path)
                    for item in row['detections']:
                        label = item['class_name']
                        category_counts[label] = category_counts.get(label, 0) + 1
            if len(groups) != 1:
                messagebox.showerror('Choose one test run',
                    'No saved detections found.' if not groups else
                    'This folder contains several runs or recordings. Choose one run folder.',
                    parent=root)
                continue
            detection_folder, recording = next(iter(groups))
            reference = BASELINE / 'true-data' / recording / 'reference_points.csv'
            if not reference.is_file():
                reference_name = filedialog.askopenfilename(
                    parent=root, title=f'Choose reference_points.csv for {recording}',
                    initialdir=str(BASELINE / 'true-data'),
                    filetypes=[('Reference points', '*.csv')])
                if not reference_name:
                    continue
                reference = Path(reference_name)
            run = find_run(detection_folder)
            measurements = next((p for p in (run / 'kalman' / 'measurements.jsonl',
                                             run / 'measurements.jsonl') if p.is_file()), None)
            category = max(category_counts, key=category_counts.get) if category_counts else CATEGORY
            output = run / 'plots' / 'position_comparison.png'
            return reference, detection_folder, output, measurements, category
    finally:
        root.destroy()


VISIBILITY_COLORS = {
    'visible': ('Visible / tracking', '#d8f1df'),
    'partial': ('Partially occluded', '#fff2ba'),
    'occluded': ('Occluded', '#f9caca'),
    'absent': ('Absent', '#e2e5e9'),
}
PIPELINE_VISIBILITY = {
    'TRACKING': 'visible',
    'PARTIALLY_OCCLUDED': 'partial',
    'FULLY_OCCLUDED': 'occluded',
}


def shade_visibility(axes, time, states):
    edges = np.r_[time[0], (time[:-1] + time[1:]) / 2, time[-1]]
    if len(time) == 1:
        edges = time[0] + np.array([-0.03, 0.03])
    used = set()
    start = 0
    for end in range(1, len(states) + 1):
        if end < len(states) and states[end] == states[start]:
            continue
        state = states[start]
        if state in VISIBILITY_COLORS:
            used.add(state)
            for axis in axes:
                axis.axvspan(edges[start], edges[end], color=VISIBILITY_COLORS[state][1],
                             linewidth=0, zorder=0)
        start = end
    return [Patch(facecolor=color, label=label)
            for state, (label, color) in VISIBILITY_COLORS.items() if state in used]


def box_center(detections, category):
    candidates = [item for item in detections if item['class_name'] == category]
    if not candidates:
        return np.full(2, np.nan)
    x0, y0, x1, y1 = max(candidates, key=lambda item: item['confidence'])['bbox_xyxy']
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2])


def project(point, calibration):
    if point is None or point[2] <= 0:
        return np.full(2, np.nan)
    pixels = np.asarray(calibration) @ point
    return pixels[:2] / pixels[2]


def read_detections(folder):
    detections = {}
    recordings = set()
    if not folder.is_dir():
        raise ValueError(f'Detection folder does not exist: {folder}')
    for path in folder.glob('*.json'):
        row = json.loads(path.read_text(encoding='utf-8'))
        if 'detections' in row and 'source_image' in row:
            detections[row['message_id'], row['bag_timestamp_ns']] = row['detections']
            recordings.add(row['recording'])
    if len(recordings) != 1:
        raise ValueError('Select a folder with saved detections from exactly one recording.')
    return detections, recordings.pop()


def read_kalman(measurements, replay):
    replay_rows = None
    if replay:
        with replay.open(encoding='utf-8', newline='') as source:
            replay_rows = {int(row['timestamp_ns']): row for row in csv.DictReader(source)}
    with measurements.open(encoding='utf-8') as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if replay_rows is not None and set(replay_rows) != {row['timestamp_ns'] for row in rows}:
        raise ValueError('The saved kalman.csv and measurements.jsonl have different filter timestamps.')
    results = {}
    raw = {}
    states = {}
    for row in sorted(rows, key=lambda item: item['timestamp_ns']):
        if replay_rows is None:
            if 'filtered_position' not in row:
                raise ValueError('No saved filtered_position. Supply the matching --kalman-csv.')
            estimate = row['filtered_position']
        else:
            saved = replay_rows[row['timestamp_ns']]
            xyz = [saved[f'filtered_{axis}'] for axis in 'xyz']
            estimate = [float(value) for value in xyz] if all(xyz) else None
        # The filename contains the source RGB timestamp, which can differ from the filter tick.
        timestamp, message_id = PureWindowsPath(row['image']).stem.rsplit('_', 1)
        key = int(message_id), int(timestamp)
        results.setdefault(key, project(estimate, row['K']))
        raw.setdefault(key, project(row['position'], row['K']))
        states.setdefault(key, PIPELINE_VISIBILITY.get(row.get('state')))
    return raw, results, states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', nargs='?', type=Path, default=None)
    parser.add_argument('detections', nargs='?', type=Path, default=None)
    parser.add_argument('output', nargs='?', type=Path, default=None)
    parser.add_argument('--category', default=CATEGORY)
    parser.add_argument('--timings', type=Path, help='Matching stage_timings.csv (normally found automatically)')
    parser.add_argument('--kalman', type=Path, default=None,
                        help='Saved measurements.jsonl with calibration')
    parser.add_argument('--kalman-csv', type=Path, default=None,
                        help='Saved kalman.csv from the same replay, if needed')
    args = parser.parse_args()
    show_plot = not sys.argv[1:]
    if show_plot:
        selection = choose_saved_run()
        if selection is None:
            return
        args.reference, args.detections, args.output, args.kalman, args.category = selection
    elif args.reference is None or args.detections is None:
        parser.error('Supply reference and detections paths, or run without arguments to open the folder picker.')
    if args.output is None:
        run_folder = find_run(args.detections)
        args.output = run_folder / 'plots' / 'position_comparison.png'
    if args.kalman_csv and not args.kalman:
        parser.error('A saved kalman.csv requires the matching measurements.jsonl.')
    if not show_plot:
        plt.switch_backend('Agg')

    detections, recording = read_detections(args.detections)
    with args.reference.open(encoding='utf-8', newline='') as source:
        references = {(int(row['message_id']), int(row['bag_timestamp_ns'])): row
                      for row in csv.DictReader(source)}
    if not references:
        raise ValueError('No saved reference annotations.')
    if not detections.keys() & references.keys():
        raise ValueError('The saved detections do not match any annotated RGB frames.')
    raw, filtered, pipeline_states = {}, {}, {}
    if args.kalman:
        raw, filtered, pipeline_states = read_kalman(args.kalman, args.kalman_csv)
        if not filtered.keys() & (detections.keys() | references.keys()):
            raise ValueError('The saved Kalman data does not match the recording.')

    keys = sorted(references.keys() | detections.keys() | filtered.keys(),
                  key=lambda key: (key[1], key[0]))
    timestamps = [key[1] for key in keys]
    reference = np.full((len(keys), 2), np.nan)
    for i, key in enumerate(keys):
        row = references.get(key)
        if row and row['visibility'] == 'visible':
            reference[i] = float(row['u_px']), float(row['v_px'])
    detected = np.array([box_center(detections.get(key, []), args.category) for key in keys])

    time = np.array([(timestamp - timestamps[0]) / 1e9 for timestamp in timestamps])
    series = [('Grounding DINO box center', detected, '#b74918', '.')]
    reference_valid = np.isfinite(reference).all(axis=1)
    if args.kalman:
        before = np.array([raw.get(key, [np.nan, np.nan]) for key in keys])
        kalman = np.array([filtered.get(key, [np.nan, np.nan]) for key in keys])
        series.extend([('Raw 3D position before Kalman', before, '#7555ac', 'x'),
                       ('Filtered 3D position after Kalman', kalman, '#007764', 'o')])
        shared = reference_valid & np.isfinite(before).all(axis=1) & np.isfinite(kalman).all(axis=1)
        error_title = f'Before vs. after Kalman on {shared.sum()} shared annotated frames'
    else:
        shared = reference_valid & np.isfinite(detected).all(axis=1)
        error_title = f'Detection error on {shared.sum()} annotated frames'
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout='constrained')
    figure.suptitle(f'{recording.split("/")[0]} | Position in the RGB image', fontsize=17)
    if any(pipeline_states.values()):
        sample_times = np.array([(key[1] - timestamps[0]) / 1e9 for key in pipeline_states])
        sample_order = np.argsort(sample_times, kind='stable')
        states = list(pipeline_states.values())
        status_handles = shade_visibility(axes, sample_times[sample_order],
                                          [states[i] for i in sample_order])
        status_source = 'Saved pipeline state'
    else:
        manual_states = [references.get(key, {}).get('visibility') for key in keys]
        status_handles = shade_visibility(axes, time, manual_states)
        status_source = 'Manual visibility'
    if status_handles:
        axes[1].legend(handles=status_handles, title=status_source, loc='best')
    for axis, coordinate in enumerate(('u [px]', 'v [px]')):
        axes[axis].plot(time, reference[:, axis], color='#243452', label='Manual reference', lw=2)
        axes[axis].set_ylabel(coordinate)
    for label, points, color, marker in series:
        valid = np.isfinite(points).all(axis=1)
        sparse = marker != '.'
        for axis in range(2):
            shown = valid if sparse else slice(None)
            axes[axis].plot(time[shown], points[shown, axis], color=color, label=label,
                            marker=marker, markersize=4,
                            linestyle='none' if sparse else '-')
        error = np.linalg.norm(points - reference, axis=1)
        errors_here = shared & valid
        stats = (f' (mean {np.mean(error[errors_here]):.1f} px, n={errors_here.sum()})'
                 if errors_here.any() else '')
        axes[2].plot(time[errors_here], error[errors_here], color=color, label=label + stats,
                     marker=marker, markersize=4)
        print(f'{label}: {valid.sum()}/{len(keys)} points{stats}')
    axes[2].set_title(error_title)
    axes[2].set_ylabel('2D error [px]')
    axes[2].set_xlabel('Recording time from first plotted frame [s]')
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(loc='best')
    axes[2].legend(loc='best')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor='white')
    print(args.output.resolve())
    timing_path = args.timings
    if timing_path is None:
        if args.kalman:
            candidate = args.kalman.parent / 'stage_timings.csv'
            if candidate.is_file():
                timing_path = candidate
        if timing_path is None:
            for parent in (args.detections, *args.detections.parents):
                candidates = (parent / 'timing' / 'stage_timings.csv', parent / 'stage_timings.csv')
                timing_path = next((p for p in candidates if p.is_file()), None)
                if timing_path is not None:
                    break
                if parent == RESULTS or parent == BASELINE:
                    break
    if timing_path is not None:
        from plot_timing import plot_timing
        plot_timing(timing_path, show=False, keep_open=show_plot)
    else:
        print('No stage_timings.csv found for this run; only the position plot was created.')
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main()
