"""Mark object centres directly on the frames used by the Python pipeline."""

import argparse
import csv
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'extracted-data'
RGB_TOPIC = '/head_front_camera/rgb/image_raw'
FIELDS = ('message_id', 'bag_timestamp_ns', 'header_timestamp_ns', 'file',
          'visibility', 'u_px', 'v_px')


def read_frames(index):
    with index.open(encoding='utf-8', newline='') as source:
        frames = [row for row in csv.DictReader(source) if row['topic'] == RGB_TOPIC]
    return sorted(frames, key=lambda row: (int(row['bag_timestamp_ns']), int(row['message_id'])))


def save_labels(path, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as target:
        writer = csv.DictWriter(target, FIELDS)
        writer.writeheader()
        writer.writerows(sorted(labels.values(), key=lambda row: (
            int(row['bag_timestamp_ns']), int(row['message_id']))))
    temporary.replace(path)


class Annotator:
    def __init__(self, window, index, frames):
        self.window = window
        self.index = index
        self.frames = frames
        self.position = 0
        self.timer = None
        self.output = ROOT / 'true-data' / index.parent.relative_to(DATA) / 'reference_points.csv'
        self.labels = {}
        if self.output.is_file():
            with self.output.open(encoding='utf-8', newline='') as source:
                self.labels = {row['message_id']: row for row in csv.DictReader(source)}

        window.title(f'Reference Points – {index.parent.name}')
        window.geometry('1000x720')
        controls = ttk.Frame(window, padding=8)
        controls.pack(fill='x')
        for label, action in (
            ('Previous', lambda: self.step(-1)),
            ('Play / Pause', self.toggle),
            ('Next', lambda: self.step(1)),
            ('Occluded (O)', lambda: self.annotate('occluded')),
            ('Absent (A)', lambda: self.annotate('absent')),
            ('Clear frame', lambda: self.annotate(None)),
            ('Clear all', self.clear_all),
        ):
            ttk.Button(controls, text=label, command=action, takefocus=False).pack(side='left', padx=4)
        self.status = ttk.Label(window, padding=8)
        self.status.pack(fill='x', side='bottom')
        self.canvas = tk.Canvas(window, bg='#222222', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', self.draw)
        self.canvas.bind('<Button-1>', self.click)
        self.canvas.bind('<Motion>', self.show_cursor_ring)
        self.canvas.bind('<Leave>', self.hide_cursor_ring)
        window.bind('<Left>', lambda event: self.step(-1))
        window.bind('<Right>', lambda event: self.step(1))
        window.bind('<space>', lambda event: self.toggle())
        window.bind('<KeyPress-o>', lambda event: self.annotate('occluded'))
        window.bind('<KeyPress-a>', lambda event: self.annotate('absent'))
        self.open_frame()

    @property
    def frame(self):
        return self.frames[self.position]

    def open_frame(self):
        with Image.open(self.index.parent / self.frame['file']) as source:
            self.image = source.convert('RGB')
        self.draw()

    def draw(self, event=None):
        width, height = self.image.size
        canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
        scale = min(canvas_width / width, canvas_height / height)
        display_width = max(1, round(width * scale))
        display_height = max(1, round(height * scale))
        self.left = (canvas_width - display_width) // 2
        self.top = (canvas_height - display_height) // 2
        self.display_size = display_width, display_height
        self.photo = ImageTk.PhotoImage(self.image.resize(self.display_size))
        self.canvas.delete('all')
        self.canvas.create_image(self.left, self.top, image=self.photo, anchor='nw')
        label = self.labels.get(self.frame['message_id'])
        state = 'Unmarked'
        if label:
            state = label['visibility'].capitalize()
            if label['visibility'] == 'visible':
                u, v = int(label['u_px']), int(label['v_px'])
                x = self.left + (u + 0.5) * display_width / width
                y = self.top + (v + 0.5) * display_height / height
                self.canvas.create_oval(x-5, y-5, x+5, y+5, outline='#64ff83', width=2)
                state += f' ({u}, {v}) px'
        self.status.config(text=f'Frame {self.position + 1}/{len(self.frames)}   |   '
                                f'{state}   |   {len(self.labels)} saved annotations')
        self.draw_cursor_ring(self.canvas.winfo_pointerx() - self.canvas.winfo_rootx(),
                              self.canvas.winfo_pointery() - self.canvas.winfo_rooty())

    def show_cursor_ring(self, event):
        self.draw_cursor_ring(event.x, event.y)

    def hide_cursor_ring(self, event=None):
        self.canvas.delete('cursor-ring')

    def draw_cursor_ring(self, x, y):
        self.hide_cursor_ring()
        width, height = self.display_size
        if self.left <= x < self.left + width and self.top <= y < self.top + height:
            bounds = (x - 10, y - 10, x + 10, y + 10)
            self.canvas.create_oval(*bounds, outline='black', width=4, tags='cursor-ring')
            self.canvas.create_oval(*bounds, outline='white', width=2, tags='cursor-ring')

    def pause(self):
        if self.timer is not None:
            self.window.after_cancel(self.timer)
            self.timer = None

    def step(self, offset):
        self.pause()
        self.position = max(0, min(self.position + offset, len(self.frames) - 1))
        self.open_frame()

    def toggle(self):
        if self.timer is not None:
            self.pause()
        else:
            self.schedule_next()

    def schedule_next(self):
        if self.position + 1 < len(self.frames):
            current = int(self.frame['bag_timestamp_ns'])
            following = int(self.frames[self.position + 1]['bag_timestamp_ns'])
            self.timer = self.window.after(max(1, round((following - current) / 1e6)), self.advance)

    def advance(self):
        self.timer = None
        self.position += 1
        self.open_frame()
        self.schedule_next()

    def click(self, event):
        x, y = event.x - self.left, event.y - self.top
        width, height = self.display_size
        if 0 <= x < width and 0 <= y < height:
            self.annotate('visible', x * self.image.width // width, y * self.image.height // height)

    def clear_all(self):
        """Clear annotations for this recording only after confirmation."""
        self.pause()
        if not self.labels:
            return
        if not messagebox.askyesno(
                'Clear all reference points?',
                f'Remove all {len(self.labels)} annotations for {self.index.parent.name}?\n'
                'This includes visible, occluded and absent labels. This cannot be undone.',
                parent=self.window, default=messagebox.NO):
            return
        try:
            save_labels(self.output, {})
        except OSError as error:
            messagebox.showerror('Could not clear annotations', str(error), parent=self.window)
            return
        self.labels = {}
        self.position = 0
        self.open_frame()

    def annotate(self, visibility, u='', v=''):
        self.pause()
        key = self.frame['message_id']
        labels = self.labels.copy()
        if visibility is None:
            labels.pop(key, None)
        else:
            labels[key] = {field: self.frame[field] for field in FIELDS[:4]}
            labels[key].update(visibility=visibility, u_px=u, v_px=v)
        try:
            save_labels(self.output, labels)
        except OSError as error:
            messagebox.showerror('Could not save annotation', str(error), parent=self.window)
            return
        self.labels = labels
        if visibility is None:
            self.draw()
        else:
            self.step(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', nargs='?', type=Path, help='Path to a recording index.csv')
    args = parser.parse_args()
    window = tk.Tk()
    window.withdraw()
    selected = args.index or filedialog.askopenfilename(
        parent=window, title='Select a recording index.csv', initialdir=DATA,
        filetypes=[('Recording index', 'index.csv')])
    if not selected:
        window.destroy()
        return
    index = Path(selected).resolve()
    try:
        if not index.is_relative_to(DATA) or index.name != 'index.csv':
            raise ValueError(f'Select an index.csv under {DATA}')
        frames = read_frames(index)
        if not frames:
            raise ValueError(f'No RGB frames found in {index}')
        Annotator(window, index, frames)
    except (OSError, ValueError, KeyError) as error:
        messagebox.showerror('Could not open recording', str(error), parent=window)
        window.destroy()
        return
    window.deiconify()
    window.mainloop()


if __name__ == '__main__':
    main()
