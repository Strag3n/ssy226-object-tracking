"""Video export and measured rates, without changing detection algorithms."""
import csv
from pathlib import Path
from statistics import median

import numpy as np
from PIL import ImageDraw


def source_timing(rows):
    """Average sample rate, not processing speed. Prefer monotonic header times."""
    if not rows:
        raise ValueError('No RGB frames selected')
    timestamps = [int(row['header_timestamp_ns']) if row['header_timestamp_ns'] else None for row in rows]
    clock = 'header'
    if any(t is None for t in timestamps) or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        timestamps = [int(row['bag_timestamp_ns']) for row in rows]
        clock = 'bag'
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError('RGB timestamps must increase to estimate playback rate')
    duration = (timestamps[-1] - timestamps[0]) / 1e9
    rate = (len(rows) - 1) / duration if duration > 0 else None
    return dict(timestamps_ns=timestamps, clock=clock, source_span_seconds=duration,
                source_fps=rate, playback_fps=rate or 15.0)


def performance_summary(latencies, wall_seconds):
    """Rates use total frame count / total time, not mean instantaneous FPS."""
    inference_seconds = sum(latencies)
    return dict(frames=len(latencies), inference_seconds=inference_seconds,
                detection_hz=len(latencies) / inference_seconds if inference_seconds else None,
                mean_detection_ms=1000 * inference_seconds / len(latencies) if latencies else None,
                median_detection_ms=1000 * median(latencies) if latencies else None,
                end_to_end_seconds=wall_seconds,
                end_to_end_hz=len(latencies) / wall_seconds if wall_seconds else None)


class TestVideo:
    """Constant-rate MP4, with source-rate playback and processing-rate overlay.

    Video playback approximates irregular source timing using the average
    sample rate. Exact timestamps and timing measurements are kept in CSV.
    """
    __test__ = False

    def __init__(self, folder: Path, size, playback_fps, filename="detections.mp4"):
        import cv2
        self.cv2 = cv2
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / filename
        self.size = tuple(size)
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*'mp4v'),
                                      float(playback_fps), self.size)
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError('MP4 encoder unavailable. Install opencv-python with video support.')
        self.csv_file = (folder / 'frame_metrics.csv').open('w', newline='', encoding='utf-8')
        self.csv = csv.DictWriter(self.csv_file, fieldnames=[
            'frame', 'message_id', 'bag_timestamp_ns', 'header_timestamp_ns',
            'source_elapsed_seconds', 'detections', 'detection_ms',
            'instantaneous_detection_hz', 'running_detection_hz'])
        self.csv.writeheader()
        self.frame_count = 0

    def write(self, image, measurement, latency, running_hz, elapsed, source_fps, frame, total):
        if image.size != self.size:
            raise ValueError('RGB image dimensions changed within the recording')
        image = image.copy()
        draw = ImageDraw.Draw(image)
        source_label = f'{source_fps:.2f}' if source_fps is not None else 'unknown'
        lines = [f'Frame {frame}/{total} | source t={elapsed:.3f}s | objects={len(measurement.detections)}',
                 f'Detection: {1000 * latency:.1f} ms | {1 / latency:.2f} Hz | running {running_hz:.2f} Hz',
                 f'Source: {source_label} FPS | playback follows source, NOT processing speed']
        for i, text in enumerate(lines):
            position = (5, 5 + i * 16)
            draw.rectangle(draw.textbbox(position, text), fill='black')
            draw.text(position, text, fill='white')
        # PIL/loader use RGB; OpenCV VideoWriter expects BGR.
        bgr = np.ascontiguousarray(np.asarray(image)[..., ::-1])
        self.writer.write(bgr)
        self.frame_count += 1
        self.csv.writerow(dict(frame=frame, message_id=measurement.message_id,
                               bag_timestamp_ns=measurement.bag_timestamp_ns,
                               header_timestamp_ns=measurement.header_timestamp_ns,
                               source_elapsed_seconds=elapsed, detections=len(measurement.detections),
                               detection_ms=latency * 1000, instantaneous_detection_hz=1 / latency,
                               running_detection_hz=running_hz))
        self.csv_file.flush()

    def close(self):
        self.writer.release()
        self.csv_file.close()
        if self.frame_count and (not self.path.exists() or self.path.stat().st_size == 0):
            raise RuntimeError(f'Video encoder produced no output: {self.path}')

def create_timing_plots(folder):
    """Create scientific timing plots from a completed test, without rerunning models."""
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    folder = Path(folder)
    with (folder / 'stage_timings.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('No timing rows to visualize')
    summary = json.loads((folder / 'timing_summary.json').read_text())
    stages = [('detection_ms','Object detection'),('segmentation_ms','Segmentation'),
              ('depth_ms','Depth model'),('position_occlusion_ms','3D + occlusion'),
              ('kalman_ms','Positional Kalman'),('load_ms','Data loading'),('export_ms','File/video export')]
    values = {key: np.array([float(row[key]) for row in rows]) for key,_ in stages}
    steps = np.array([int(row['step']) for row in rows])
    source_time = (np.array([int(row['timestamp_ns']) for row in rows],dtype=np.int64)
                   - int(rows[0]['timestamp_ns'])) / 1e9
    colors = ['#2563eb','#c2410c','#0f766e','#7c3aed','#be123c','#64748b','#a16207']
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2,2,figsize=(14,9),layout='constrained')
    fig.suptitle(f"CPU pipeline timings | {summary['recording']} | {len(rows)} processed steps",fontsize=15)
    ax=axes[0,0]
    means=np.array([values[key].mean()/1000 for key,_ in stages])
    ax.barh([name for _,name in stages],means,color=colors)
    ax.set_xscale('log')
    ax.invert_yaxis()
    for i,value in enumerate(means):
        label=f'{value:.2f} s' if value>=.1 else f'{value*1000:.3f} ms'
        ax.text(value*1.15,i,label,va='center')
    ax.set_xlim(max(min(means[means>0])*.5,1e-7),max(means)*4)
    ax.set_xlabel('Mean time per processed step (seconds, logarithmic scale)')
    ax.set_title('Where computation time is spent')
    ax.grid(axis='x',alpha=.2)

    ax=axes[0,1]
    ax.stackplot(steps,*[values[key]/1000 for key,_ in stages],labels=[name for _,name in stages],colors=colors)
    budget=1/summary['scheduled_rate_hz']
    ax.axhline(budget,color='black',linestyle='--',linewidth=1,label=f'{summary["scheduled_rate_hz"]:g} Hz budget ({budget:.1f} s)')
    ax.set(xlabel='Processed step',ylabel='Stage time (seconds)',title='Time allocation for every step')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8,loc='upper right',ncol=2)

    ax=axes[1,0]
    for mode,marker,color in [('initialize','s','#7c3aed'),('update','o','#2563eb'),
                              ('predict_occluded','^','#c2410c'),('predict_no_position','v','#0f766e'),
                              ('waiting','x','#64748b'),('reset','D','#be123c')]:
        selected=np.array([r['kalman_mode']==mode for r in rows])
        if selected.any():
            ax.scatter(steps[selected],values['kalman_ms'][selected],label=mode,s=23,marker=marker,color=color)
    ax.set(xlabel='Processed step',ylabel='Filter call time (milliseconds)',title='Kalman timing, separated by operation')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)

    ax=axes[1,1]
    processing=np.cumsum([float(row['end_to_end_ms'])/1000 for row in rows])
    ax.plot(source_time,processing,label='Cumulative processing time',color='#2563eb',linewidth=2)
    ax.plot(source_time,source_time,label='Real-time reference',color='black',linestyle='--')
    ax.set(xlabel='Elapsed recording time (seconds)',ylabel='Elapsed processing time (seconds)',title='Offline work versus recording duration')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.supxlabel('All processed steps shown, including the first cold frame. Model loading excluded. Depth may be skipped in occluded frames.',fontsize=9)
    plot_folder = (folder.parent if folder.name == 'timing' else folder) / 'plots'
    plot_folder.mkdir(parents=True, exist_ok=True)
    timing_path=plot_folder/'timing_plots.png'
    fig.savefig(timing_path,dpi=160)
    plt.close(fig)
    paths=[timing_path]

    measurement_path=folder/'measurements.jsonl'
    if not measurement_path.exists() and folder.name == 'timing':
        measurement_path=folder.parent/'kalman'/'measurements.jsonl'
    if measurement_path.exists():
        data=[json.loads(line) for line in measurement_path.read_text().splitlines() if line.strip()]
        if data and any(r.get('filtered_position') is not None for r in data):
            t=(np.array([r['timestamp_ns'] for r in data],dtype=np.int64)-data[0]['timestamp_ns'])/1e9
            measured=np.array([r['position'] if r['position'] is not None else [np.nan]*3 for r in data])
            filtered=np.array([r['filtered_position'] if r.get('filtered_position') is not None else [np.nan]*3 for r in data])
            fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
            for axis,label in enumerate('XYZ'):
                axes[axis].plot(t,measured[:,axis],'o',markersize=3,color='#c2410c',label='3D measurement')
                axes[axis].plot(t,filtered[:,axis],color='#2563eb',label='Kalman estimate')
                axes[axis].set_ylabel(f'Camera {label} (m)')
                axes[axis].grid(alpha=.2)
            axes[0].legend()
            axes[-1].set_xlabel('Elapsed recording time (seconds)')
            fig.suptitle('Measured and filtered camera-frame positions | no ground-truth accuracy evaluation')
            fig.supxlabel('Camera/robot motion is not compensated. RGB-cloud registration remains an unverified original assumption.',fontsize=9)
            position_path=plot_folder/'position_plots.png'
            fig.savefig(position_path,dpi=160)
            plt.close(fig)
            paths.append(position_path)
    return paths


if __name__ == '__main__':
    root=Path(__file__).resolve().parents[1]/'results'
    folders=sorted({p.parent for p in root.rglob('stage_timings.csv')}, key=lambda folder: str(folder))
    if not folders:
        raise SystemExit('No timed pipeline tests found. Run start.py and choose test type 2 first.')
    for number,folder in enumerate(folders,1):
        print(f'{number}. {folder.relative_to(root)}')
    answer=input('Choose test number (Enter = latest): ').strip()
    selected=folders[-1] if not answer else folders[int(answer)-1]
    for plot in create_timing_plots(selected):
        print(f'Plot saved: {plot}')
