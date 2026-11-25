"""Extract recorded camera data without ROS.

Edit INPUT_FOLDER and OUTPUT_FOLDER below, then run this file.
Flow: input folder -> recording subfolders -> .db3 -> CDR -> PNG/NPY/JSON.
See README.md for a complete example using apple-clutter.
"""
import csv
import json
from pathlib import Path
import sqlite3
import struct

import numpy as np
from PIL import Image


# ==================== EDIT YOUR PATHS HERE ====================
# The project folder is the parent of this script's folder.
# This works regardless of the terminal's current working directory.
PROJECT_FOLDER = Path(__file__).resolve().parent.parent
INPUT_FOLDER = PROJECT_FOLDER.parent / 'pose-estimation-data-new'
OUTPUT_FOLDER = PROJECT_FOLDER / 'extracted-data'
# Use a new OUTPUT_FOLDER name if that folder already exists.
# Set to 1 for a small trial, or None to extract every message.
LIMIT_PER_TOPIC = None
# =============================================================


class CDR:
    """CDR1 reader; alignment is relative to the encapsulated payload."""
    def __init__(self, data):
        if data[:2] not in (b'\x00\x00', b'\x00\x01'):
            raise ValueError('Only plain CDR1 serialization is supported')
        self.data = memoryview(data)[4:]
        self.pos = 0
        self.endian = '<' if data[1] else '>'

    def number(self, fmt):
        size = struct.calcsize(fmt)
        self.pos += (-self.pos) % size
        value = struct.unpack_from(self.endian + fmt, self.data, self.pos)[0]
        self.pos += size
        return value

    def raw(self, length):
        end = self.pos + length
        if end > len(self.data):
            raise ValueError('Truncated CDR payload')
        value = bytes(self.data[self.pos:end])
        self.pos = end
        return value

    def string(self):
        value = self.raw(self.number('I'))
        if not value or value[-1] != 0:
            raise ValueError('Invalid CDR string')
        return value[:-1].decode('utf-8')

    def header(self):
        sec, nano = self.number('i'), self.number('I')
        return {'stamp_ns': sec * 1_000_000_000 + nano, 'frame_id': self.string()}

    def floats(self, count):
        return [self.number('d') for _ in range(count)]

    def finish(self):
        if len(self.data) - self.pos > 3:
            raise ValueError(f'Unparsed payload: {len(self.data) - self.pos} bytes')


def decode(data, kind):
    """Unpack binary ROS fields in the message type's defined order.

    This reads existing measurements. It does not compute new measurements.
    Both RGB and depth use Image messages; encoding distinguishes their formats.
    """
    r = CDR(data)
    if kind == 'tf2_msgs/msg/TFMessage':
        result = {'transforms': []}
        for _ in range(r.number('I')):
            result['transforms'].append({**r.header(), 'child_frame_id': r.string(),
                                         'translation': r.floats(3), 'quaternion_xyzw': r.floats(4)})
    else:
        result = r.header()
        result.update(height=r.number('I'), width=r.number('I'))
        if kind == 'sensor_msgs/msg/Image':
            result.update(encoding=r.string(), is_bigendian=r.number('B'), step=r.number('I'))
            result['data'] = r.raw(r.number('I'))
        elif kind == 'sensor_msgs/msg/CameraInfo':
            result.update(distortion_model=r.string(), D=r.floats(r.number('I')),
                          K=r.floats(9), R=r.floats(9), P=r.floats(12),
                          binning_x=r.number('I'), binning_y=r.number('I'))
            result['roi'] = dict(zip(('x_offset', 'y_offset', 'height', 'width'),
                                     [r.number('I') for _ in range(4)]))
            result['roi']['do_rectify'] = bool(r.number('B'))
        elif kind == 'sensor_msgs/msg/PointCloud2':
            result['fields'] = [dict(name=r.string(), offset=r.number('I'),
                                     datatype=r.number('B'), count=r.number('I'))
                                for _ in range(r.number('I'))]
            result.update(is_bigendian=r.number('B'), point_step=r.number('I'), row_step=r.number('I'))
            result['data'] = r.raw(r.number('I'))
            result['is_dense'] = bool(r.number('B'))
        else:
            raise ValueError(f'Unsupported message type: {kind}')
    r.finish()
    return result


def image_array(msg):
    """Convert an image buffer into an (H, W, 3) or (H, W) array.

    encoding defines the data type and channel count. step is bytes per row,
    including padding. is_bigendian defines byte order.
    These properties must be respected instead of simply reshaping raw bytes.
    """
    formats = {'rgb8': ('u1', 3), 'bgr8': ('u1', 3), 'mono8': ('u1', 1),
               '16UC1': ('u2', 1), 'mono16': ('u2', 1), '32FC1': ('f4', 1)}
    dtype, channels = formats[msg['encoding']]
    dtype = np.dtype(('>' if msg['is_bigendian'] else '<') + dtype)
    h, w, step = msg['height'], msg['width'], msg['step']
    if step < w * channels * dtype.itemsize or len(msg['data']) != h * step:
        raise ValueError('Invalid image dimensions/stride')
    shape = (h, w, channels) if channels > 1 else (h, w)
    strides = (step, dtype.itemsize * channels, dtype.itemsize) if channels > 1 else (step, dtype.itemsize)
    return np.ndarray(shape, dtype=dtype, buffer=msg['data'], strides=strides).copy()


def cloud_array(msg):
    """Read the recorded point cloud as a structured NumPy array.

    fields describes each point: field names, types, counts and byte offsets.
    point_step is bytes per point; row_step is bytes per row.
    All fields, including color and invalid values (NaN), are preserved.
    Access cloud['x'][row, column] when the x field exists.
    This reads an existing cloud rather than generating one from depth.
    """
    types = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
    endian = '>' if msg['is_bigendian'] else '<'
    fields = msg['fields']
    dtype = np.dtype({'names': [f['name'] for f in fields],
                      'formats': [(np.dtype(endian + types[f['datatype']]), (f['count'],))
                                  if f['count'] != 1 else np.dtype(endian + types[f['datatype']]) for f in fields],
                      'offsets': [f['offset'] for f in fields], 'itemsize': msg['point_step']})
    if len(msg['data']) != msg['height'] * msg['row_step'] or msg['row_step'] < msg['width'] * msg['point_step']:
        raise ValueError('Invalid point cloud dimensions/stride')
    return np.ndarray((msg['height'], msg['width']), dtype=dtype, buffer=msg['data'],
                      strides=(msg['row_step'], msg['point_step'])).copy()


def save_metadata(msg, stem):
    """Save the header and format description beside the data, without raw bytes."""
    metadata = {key: value for key, value in msg.items() if key != 'data'}
    stem.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')


def extract_rgb(msg, stem):
    """RGB: byte buffer -> uint8 image -> lossless PNG.

    BGR channels are reordered to RGB. mono8 is saved as grayscale.
    No normalization, cropping or resizing is performed.
    """
    array = image_array(msg)
    if msg['encoding'] == 'bgr8':
        array = array[..., ::-1]
    path = stem.with_suffix('.png')
    Image.fromarray(array).save(path)
    save_metadata(msg, stem)
    return path


def extract_depth(msg, stem):
    """Depth: byte buffer -> one raw depth value per pixel -> NPY.

    16UC1/mono16 stays uint16; 32FC1 stays float32.
    Values are not converted to meters. Zeros and NaNs are preserved.
    NPY preserves precision for subsequent Python processing.
    """
    array = image_array(msg)
    path = stem.with_suffix('.npy')
    np.save(path, array, allow_pickle=False)
    save_metadata(msg, stem)
    return path


def extract_point_cloud(msg, stem):
    """PointCloud2: byte buffer + field definitions -> structured NPY.

    Preserve the cloud's coordinate frame and organized (H, W) shape.
    No filtering, coordinate transformation or projection is performed.
    """
    path = stem.with_suffix('.npy')
    np.save(path, cloud_array(msg), allow_pickle=False)
    save_metadata(msg, stem)
    return path


def extract_camera_info(msg, stem):
    """Calibration -> JSON: K, R, P, distortion, image dimensions and ROI."""
    save_metadata(msg, stem)
    return stem.with_suffix('.json')


def extract_transforms(msg, stem):
    """TF -> JSON: timestamp, parent/child frames, translation and quaternion."""
    save_metadata(msg, stem)
    return stem.with_suffix('.json')


def save_message(msg, kind, stem):
    """Select the exporter for the decoded message type and image encoding."""
    if kind == 'sensor_msgs/msg/Image':
        if msg['encoding'] in ('rgb8', 'bgr8', 'mono8'):
            return extract_rgb(msg, stem)
        return extract_depth(msg, stem)
    if kind == 'sensor_msgs/msg/PointCloud2':
        return extract_point_cloud(msg, stem)
    if kind == 'sensor_msgs/msg/CameraInfo':
        return extract_camera_info(msg, stem)
    if kind == 'tf2_msgs/msg/TFMessage':
        return extract_transforms(msg, stem)
    raise ValueError(f'Unsupported message type: {kind}')


def extract(source, target, limit=None):
    """Read source recordings, export measurements to target, and write an index.

    INPUT_FOLDER is passed here as source when this file is run.
    rglob searches all its subfolders, for example:
    pose-estimation-data-new/apple-clutter/apple-clutter_0.db3
    Each database is opened read-only and its messages are streamed one by one.
    """
    source = source.resolve()
    target = target.resolve()
    # Search INPUT_FOLDER, including all recording subfolders.
    databases = sorted(source.rglob('*.db3')) if source.is_dir() else [source]
    if not databases or not all(p.is_file() for p in databases):
        raise ValueError('No SQLite recordings found')
    if target.exists():
        raise FileExistsError(f'Choose a new output directory: {target}')
    print(f'Input recordings: {source}', flush=True)
    print(f'Found {len(databases)} .db3 files', flush=True)
    print(f'Export destination: {target}', flush=True)
    target.mkdir(parents=True)
    summary = []
    for db in databases:
        # Each database gets its own directory, including split recordings.
        folder = target / db.parent.name / db.stem
        folder.mkdir(parents=True)
        counts = {}
        print(f'Reading: {db}', flush=True)
        # SQLite reads the actual .db3 file under the selected input folder.
        with sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True) as conn:
            topics = {row[0]: row[1:] for row in conn.execute('SELECT id,name,type,serialization_format FROM topics')}
            inventory = [dict(topic=name, type=kind, count=n) for name, kind, n in conn.execute(
                'SELECT t.name,t.type,count(m.id) FROM topics t LEFT JOIN messages m ON m.topic_id=t.id GROUP BY t.id')]
            with (folder / 'index.csv').open('w', newline='', encoding='utf-8') as index:
                writer = csv.DictWriter(index, fieldnames=['message_id', 'topic', 'type', 'bag_timestamp_ns', 'header_timestamp_ns', 'frame_id', 'file'])
                writer.writeheader()
                for mid, tid, timestamp, data in conn.execute('SELECT id,topic_id,timestamp,data FROM messages ORDER BY timestamp,id'):
                    if limit is not None and counts.get(tid, 0) >= limit:
                        continue
                    name, kind, serialization = topics[tid]
                    if serialization != 'cdr':
                        raise ValueError(f'Unsupported serialization: {serialization}')
                    # 1. Unpack the database's binary CDR message into named fields.
                    msg = decode(data, kind)
                    topic_folder = folder / f'topic_{tid:03d}'
                    topic_folder.mkdir(exist_ok=True)
                    stem = topic_folder / f'{timestamp}_{mid}'
                    # 2. Export RGB, depth, point cloud, calibration or TF.
                    path = save_message(msg, kind, stem)
                    # 3. Record the topic, output file and both timestamps.
                    writer.writerow(dict(message_id=mid, topic=name, type=kind, bag_timestamp_ns=timestamp,
                                         header_timestamp_ns=msg.get('stamp_ns', ''), frame_id=msg.get('frame_id', ''),
                                         file=path.relative_to(folder).as_posix()))
                    counts[tid] = counts.get(tid, 0) + 1
        record = dict(source=str(db.resolve()), topics=inventory,
                      extracted={topics[tid][0]: n for tid, n in counts.items()})
        (folder / 'summary.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        summary.append(record)
        print(f'{db.parent.name}/{db.name}: {sum(counts.values())} messages', flush=True)
    (target / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    if LIMIT_PER_TOPIC is not None and LIMIT_PER_TOPIC < 1:
        raise ValueError('LIMIT_PER_TOPIC must be positive or None')
    extract(source=INPUT_FOLDER, target=OUTPUT_FOLDER, limit=LIMIT_PER_TOPIC)
