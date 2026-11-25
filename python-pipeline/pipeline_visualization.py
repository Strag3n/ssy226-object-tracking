"""Shared visual overlays for live offline tests and saved-video rebuilding."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def font(size):
    for name in ('DejaVuSans.ttf', 'arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def pipeline_overlay(image, mask, measurement, stage=None):
    """Keep detections on the input image; add mask and camera-projected positions."""
    view = image.convert('RGB').copy()
    if mask is not None:
        binary = Image.fromarray(np.asarray(mask, dtype=np.uint8)*255).resize(view.size, Image.Resampling.NEAREST)
        tint = Image.new('RGB', view.size, '#22c55e')
        view = Image.composite(Image.blend(view, tint, .28), view, binary)
        # Bright boundary remains visible even on green objects such as apples.
        edge = np.asarray(binary.filter(ImageFilter.MaxFilter(5))) != np.asarray(binary.filter(ImageFilter.MinFilter(3)))
        view = Image.composite(Image.new('RGB', view.size, '#39ff14'), view,
                               Image.fromarray(edge.astype(np.uint8)*255))
    draw = ImageDraw.Draw(view)
    K = np.asarray(measurement['K'], dtype=float)
    status = []
    for key, label, color in [('position', '3D measurement', '#fb923c'),
                               ('filtered_position', 'Kalman', '#facc15')]:
        point = measurement.get(key)
        if point is None or not np.isfinite(point).all() or point[2] <= 0:
            status.append(label + ': unavailable')
            continue
        projected = K @ np.asarray(point)
        u, v = projected[:2] / projected[2]
        if not (0 <= u < view.width and 0 <= v < view.height):
            status.append(label + ': outside image')
            continue
        if key == 'filtered_position':
            draw.ellipse((u-11,v-11,u+11,v+11), outline='black', width=5)
            draw.ellipse((u-10,v-10,u+10,v+10), outline=color, width=3)
            draw.line((u-17,v,u+17,v), fill=color, width=2)
            draw.line((u,v-17,u,v+17), fill=color, width=2)
        else:
            draw.ellipse((u-4,v-4,u+4,v+4), fill=color, outline='black')
    labels = [('SAM2 mask' if mask is not None else 'SAM2 mask: unavailable', '#22c55e'),
              ('3D measurement', '#fb923c'), ('Kalman estimate', '#facc15')]
    # Put information above the image, preserving every source-image pixel.
    canvas = Image.new('RGB', (view.width, view.height + 132), '#0f172a')
    canvas.paste(view, (0, 132))
    view = canvas
    draw = ImageDraw.Draw(view)
    y = 64
    x = 8
    for label,color in labels:
        draw.rectangle((x,y+3,x+9,y+12),fill=color)
        draw.text((x+14,y),label,font=font(13),fill='white')
        x += int(draw.textlength(label,font=font(13))) + 29
    state = measurement.get('state','unknown')
    mode = measurement.get('kalman_mode','unknown')
    draw.text((8,y+22),f'State: {state} | Kalman: {mode}',font=font(13),fill='white')
    if stage:
        text = ' | '.join(f'{label} {float(stage.get(key,0))/1000:.3f}s' for key,label in
                         [('detection_ms','DINO'),('segmentation_ms','SAM'),('depth_ms','Depth'),('kalman_ms','KF')])
    else:
        text = ' | '.join(status)
    draw.text((8,y+44),text,font=font(12),fill='#cbd5e1')
    return view
