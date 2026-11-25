"""Rebuild full-pipeline videos from saved results; no tracking is rerun."""
import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

BASELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASELINE / 'python-pipeline'))
from pipeline_visualization import pipeline_overlay


def rebuild(run, recompute_missing=False):
    rows = [json.loads(line) for line in (run/'kalman'/'measurements.jsonl').read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'No measurements: {run}')
    summary = json.loads((run/'timing'/'timing_summary.json').read_text())
    with (run/'timing'/'stage_timings.csv').open(newline='') as f:
        timings = {int(r['timestamp_ns']): r for r in csv.DictReader(f)}
    # Build to a temporary MP4 so failed exports do not replace a working video.
    output = run/'video'/'pipeline.mp4'
    temporary = output.with_name('pipeline.building.mp4')
    segmenter = None
    reconstructed = 0
    writer = None
    count = 0
    try:
        for row in rows:
            with Image.open(row['image']) as source:
                view = source.convert('RGB')
            draw = ImageDraw.Draw(view)
            for detection in row.get('detections', []):
                box = detection['bbox_xyxy']
                draw.rectangle(box, outline='#60a5fa', width=2)
                draw.text((box[0]+2, max(0,box[1]-13)), f"{detection['class_name']} {detection['confidence']:.2f}", fill='#60a5fa')
            message_id = Path(row['image']).stem.rsplit('_',1)[1]
            stem = f"{row['timestamp_ns']}_{message_id}.npy"
            mask_path = run/'segmentation'/stem
            recovered = run/'segmentation-reconstructed'/stem
            mask = None
            if mask_path.is_file():
                mask = np.load(mask_path, allow_pickle=False)
            elif recovered.is_file():
                mask = np.load(recovered, allow_pickle=False)
            elif recompute_missing and row.get('sam_score') is not None:
                if segmenter is None:
                    from object_pose_estimator.offline_runner import make_segmenter
                    segmenter = make_segmenter(summary.get('device','cpu'))
                with Image.open(row['image']) as original:
                    mask, _, _ = segmenter.segment(np.asarray(original.convert('RGB')), row['detections'])
                if mask is not None:
                    recovered.parent.mkdir(exist_ok=True)
                    np.save(recovered, mask, allow_pickle=False)
                    reconstructed += 1
            stage = timings.get(int(row['timestamp_ns']), {})
            view = pipeline_overlay(view, mask, row, stage)
            ImageDraw.Draw(view).text((8,8),f"Step {count+1}/{len(rows)} | t={(row['timestamp_ns']-rows[0]['timestamp_ns'])/1e9:.2f}s",fill='white',stroke_width=2,stroke_fill='black')
            if writer is None:
                writer = cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),float(summary['scheduled_rate_hz']),view.size)
                if not writer.isOpened():
                    raise RuntimeError('Video encoder unavailable')
            writer.write(cv2.cvtColor(np.asarray(view),cv2.COLOR_RGB2BGR))
            count += 1
        writer.release()
        writer = None
        cap = cv2.VideoCapture(str(temporary))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = cap.read()
        cap.release()
        if not ok or frames != len(rows):
            raise RuntimeError(f'Video validation failed: {frames} frames vs {len(rows)}')
        temporary.replace(output)
        if reconstructed:
            (run/'segmentation-reconstructed'/'provenance.json').write_text(json.dumps({
                'note':'Masks recomputed for visualisation from saved RGB/detections; not original timed masks.',
                'sam_model':summary.get('sam_model'),'new_masks':reconstructed},indent=2))
        summary['video'] = str(output.resolve())
        (run/'timing'/'timing_summary.json').write_text(json.dumps(summary,indent=2))
        print(f'OK: {run.name} | {count} frames | {reconstructed} reconstructed masks',flush=True)
    finally:
        if writer is not None:
            writer.release()
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',nargs='?',type=Path)
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--recompute-missing',action='store_true')
    args=parser.parse_args()
    runs=sorted(p.parent.parent for p in (BASELINE/'results').rglob('kalman/measurements.jsonl')) if args.all else [args.run]
    if not args.all and args.run is None:
        parser.error('Supply a run folder or --all')
    for run in runs:
        rebuild(run,args.recompute_missing)


if __name__=='__main__':
    main()
