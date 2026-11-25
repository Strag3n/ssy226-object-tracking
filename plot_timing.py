"""Choose saved stage_timings.csv and plot processing time as coloured areas."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASELINE = Path(__file__).resolve().parent
STAGES = [
    ('detection_ms', 'Object detection', '#3b82f6'),
    ('segmentation_ms', 'Segmentation', '#14b8a6'),
    ('depth_ms', 'Depth model', '#a78bfa'),
    ('position_occlusion_ms', '3D + occlusion', '#fbbf24'),
    ('kalman_ms', 'Kalman filter', '#fb7185'),
]


def choose_file():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    try:
        return filedialog.askopenfilename(
            parent=root, title='Choose stage_timings.csv from a test run',
            initialdir=str(BASELINE / 'results'),
            filetypes=[('Timing data', 'stage_timings.csv')])
    finally:
        root.destroy()


def plot_timing(path, show=True, keep_open=False):
    with path.open(encoding='utf-8-sig', newline='') as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError('This test has no saved timing rows.')
    steps = np.array([int(row['step']) for row in rows])
    values = np.array([[float(row[key]) / 1000 for row in rows] for key, _, _ in STAGES])
    total = values.sum(axis=0)
    labels = [name for _, name, _ in STAGES]
    colors = [color for _, _, color in STAGES]
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, (ax, bars) = plt.subplots(2, 1, figsize=(12, 8),
            gridspec_kw={'height_ratios': [3, 1]}, layout='constrained')
        fig.set_facecolor('#f8fafc')
        fig.suptitle('Where does the processing time go?', fontsize=21, fontweight='bold', x=.06, ha='left')
        ax.set_title(f'{path.parent.name}  |  {len(rows)} saved processing steps',
                     loc='left', fontsize=11, color='#475569', pad=14)
        ax.stackplot(steps, *values, labels=labels, colors=colors, alpha=.92)
        ax.plot(steps, total, color='#0f172a', linewidth=1.4, label='Total of shown stages')
        ax.set(xlabel='Processing step', ylabel='Processing time per step (seconds)', ylim=(0, None))
        if len(steps) == 1:
            ax.set_xlim(steps[0]-.5, steps[0]+.5)
        ax.grid(axis='y', alpha=.16)
        ax.set_axisbelow(True)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
        means = values.mean(axis=1)
        left = 0
        for name, color, mean in zip(labels, colors, means):
            bars.barh([0], [mean], left=left, color=color, height=.45)
            if mean > means.sum()*.065:
                bars.text(left+mean/2, 0, f'{mean:.2f} s', ha='center', va='center', fontsize=10)
            left += mean
        bars.axvline(total.mean(), color='#0f172a', linewidth=1.2)
        bars.set(yticks=[], xlabel='Mean stage time per step (seconds)', ylim=(-.6, .6))
        bars.set_title(f'Mean stage total {total.mean():.2f} s  |  Median {np.median(total):.2f} s  |  95th percentile {np.percentile(total,95):.2f} s',
                       loc='left', fontsize=11)
        bars.grid(axis='x', alpha=.16)
        bars.set_axisbelow(True)
        fig.supxlabel('Five processing stages only; loading, saving and overhead excluded. All saved steps included. Skipped stages contribute zero.',
                      fontsize=9, color='#475569')
        run = path.parent.parent if path.parent.name == 'timing' else path.parent
        output = run / 'plots' / 'processing_time.png'
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
        print(f'Saved: {output}')
        if show:
            plt.show()
        if not keep_open:
            plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('timings', nargs='?', type=Path)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    selected = args.timings or choose_file()
    if selected:
        plot_timing(Path(selected), show=not args.no_show)


if __name__ == '__main__':
    main()
