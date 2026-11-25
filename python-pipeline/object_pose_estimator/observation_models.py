"""Grounding DINO detection and SAM2 segmentation for recorded RGB frames."""

from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from start import Measurement

PILImage = Image
# Same values as the original constants.py; no dependency on reference files.
SEGMENTATION_WIDTH = 320
SEGMENTATION_HEIGHT = 240


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GroundingDinoCore:
    def __init__(self, gdino_model='IDEA-Research/grounding-dino-tiny',
                 box_thresh=0.25, device=None):
        # Load ML packages only when constructing the model.
        global torch, cv2, AutoProcessor, AutoModelForZeroShotObjectDetection
        import torch
        import cv2
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self._device = device or _pick_device()
        self._box_thresh = box_thresh
        self._processor = AutoProcessor.from_pretrained(gdino_model)
        self._gdino = AutoModelForZeroShotObjectDetection.from_pretrained(
            gdino_model).to(self._device)
        self._gdino.eval()

    def detect(self, image_rgb, categories):
        # The original callback performs this resize before detection/SAM2.
        image = np.asarray(image_rgb, dtype=np.uint8)
        image = cv2.resize(image, (SEGMENTATION_WIDTH, SEGMENTATION_HEIGHT))
        detections = []
        for category in categories:
            for box, score in self._gdino_detect(image, category):
                detections.append(dict(category=category, box=box, score=score))
        return image, detections

    def _gdino_detect(self, image_rgb: np.ndarray, category: str):
        """Run Grounding-DINO on image_rgb for category text prompt.

        Returns a list of ((x1,y1,x2,y2), score) tuples, one per box that
        passes the confidence threshold (empty list if none do).
        """
        H, W = image_rgb.shape[:2]
        pil_img = PILImage.fromarray(image_rgb)
        text = category.rstrip('.') + '.'

        inputs = self._processor(images=pil_img, text=text, return_tensors='pt')
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._gdino(**inputs)

        scores = outputs.logits[0].sigmoid().max(dim=-1).values.cpu().numpy()
        pred_boxes = outputs.pred_boxes[0].cpu().numpy()

        keep = scores >= self._box_thresh
        scores = scores[keep]
        pred_boxes = pred_boxes[keep]

        detections = []
        for score, (cx, cy, bw, bh) in zip(scores, pred_boxes):
            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)
            detections.append(((x1, y1, x2, y2), float(score)))
        return detections



class ObjectDetector:
    """Offline adapter around the original Grounding DINO detection method.

    The core returns boxes in the original pipeline's resized 320x240 image.
    Keep those boxes and also map them back to full-resolution image pixels
    for visualization. Original image arrays and source metadata stay intact.
    """
    def __init__(self, model_id='IDEA-Research/grounding-dino-tiny',
                 confidence=0.25, device=None, categories=None):
        if not 0 <= confidence <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        if not categories or any(not isinstance(c, str) or not c.strip() for c in categories):
            raise ValueError('Set OBJECT_CATEGORIES to a nonempty list of text prompts')
        self.core = GroundingDinoCore(model_id, confidence, device)
        self.model_id = model_id
        self.confidence = confidence
        self.device = self.core._device
        self.categories = list(categories)

    def configure_cpu(self, short_edge=240, threads=None):
        """Optional speed/accuracy tradeoff; None preserves original preprocessing."""
        import torch
        if self.device != 'cpu':
            return
        if short_edge is not None:
            if short_edge < 128:
                raise ValueError('Use an input short edge of at least 128 pixels')
            self.core._processor.image_processor.size = {
                'shortest_edge': short_edge, 'longest_edge': round(short_edge * 4 / 3)}
        if threads is not None:
            if threads < 1:
                raise ValueError('CPU_THREADS must be positive or None')
            torch.set_num_threads(threads)
        print(f'Processor image size: {self.core._processor.image_processor.size}; CPU threads: {torch.get_num_threads()}', flush=True)

    def __call__(self, measurement: Measurement) -> Measurement:
        if measurement.kind != 'rgb':
            return measurement
        rgb = measurement.data
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Object detection expects an H x W x 3 uint8 RGB array')
        resized, original_detections = self.core.detect(rgb, self.categories)
        height, width = rgb.shape[:2]
        resized_height, resized_width = resized.shape[:2]
        scale = (width / resized_width, height / resized_height,
                 width / resized_width, height / resized_height)
        detections = []
        for detection in original_detections:
            box = detection['box']
            detections.append(dict(
                class_name=detection['category'], confidence=float(detection['score']),
                bbox_xyxy=[value * factor for value, factor in zip(box, scale)],
                bbox_xyxy_segmentation=list(box),
                segmentation_image_size=[resized_width, resized_height]))
        return replace(measurement, detections=tuple(detections))

    def synchronize(self):
        """Wait for GPU work before/after wall-clock inference timing."""
        import torch
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize(self.device)
        elif str(self.device).startswith('mps'):
            torch.mps.synchronize()

    def annotate(self, measurement: Measurement) -> Image.Image:
        """Draw on a copy, leaving the source RGB image unchanged."""
        if measurement.kind != 'rgb' or measurement.detections is None:
            raise ValueError('Run detection on an RGB measurement before drawing')
        image = Image.fromarray(measurement.data)
        draw = ImageDraw.Draw(image)
        for detection in measurement.detections:
            box = detection['bbox_xyxy']
            draw.rectangle(box, outline='lime', width=3)
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            x, y = max(0, box[0]), max(0, box[1] - 14)
            text_box = draw.textbbox((x, y), label)
            draw.rectangle(text_box, fill='black')
            draw.text((x, y), label, fill='lime')
        return image

    def save_result(self, measurement: Measurement, output_folder: Path, *, nest_recording=True) -> Path:
        """Write an annotated PNG and a JSON result, even for zero detections."""
        image = self.annotate(measurement)
        folder = output_folder / measurement.recording if nest_recording else output_folder
        folder.mkdir(parents=True, exist_ok=True)
        stem = folder / f'{measurement.bag_timestamp_ns}_{measurement.message_id}'
        image_path = stem.with_suffix('.png')
        image.save(image_path)
        record = dict(source_image=str(measurement.path), topic=measurement.topic,
                      recording=measurement.recording, message_id=measurement.message_id,
                      bag_timestamp_ns=measurement.bag_timestamp_ns,
                      header_timestamp_ns=measurement.header_timestamp_ns,
                      frame_id=measurement.frame_id, image_width=image.width,
                      image_height=image.height, model=self.model_id,
                      confidence_threshold=self.confidence, device=self.device,
                      object_categories=self.categories, bbox_format='xyxy_original_image_pixels',
                      segmentation_bbox_format='xyxy_resized_image_pixels_unclipped',
                      detections=list(measurement.detections))
        stem.with_suffix('.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        return image_path


class Segmenter:
    def __init__(self, model_id='facebook/sam2.1-hiera-base-plus', device='cpu',
                 width=320, height=240, min_score=0.20, min_mask_area=0):
        import torch
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as error:
            raise ImportError('Install SAM2 as described in README.md before preparing 3D measurements.') from error
        self.torch = torch
        self.sam = SAM2ImagePredictor.from_pretrained(model_id, device=device)
        self.width, self.height = width, height
        self.min_score, self.min_mask_area = min_score, min_mask_area

    def segment(self, rgb, detections):
        import cv2
        image = np.asarray(rgb, dtype=np.uint8)
        image = cv2.resize(image, (self.width, self.height))
        self.sam.set_image(image)
        best_object = None
        for detection in detections:
            box = detection['bbox_xyxy_segmentation']
            if detection.get('segmentation_image_size') != [self.width, self.height]:
                raise ValueError('Saved box dimensions do not match the original segmentation input')
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            with self.torch.inference_mode():
                masks, sam_scores, _ = self.sam.predict(
                    point_coords=np.array([[cx, cy]], dtype=np.float32),
                    point_labels=np.array([1]),
                    box=np.array([x1, y1, x2, y2], dtype=np.float32),
                    multimask_output=True,
                )
            best = int(np.argmax(sam_scores))
            mask = masks[best].astype(np.uint8)
            score = float(sam_scores[best])
            if best_object is None or score > best_object[0]:
                best_object = (score, mask)
        if best_object is None or best_object[0] < self.min_score:
            return None, None, None
        score, mask = best_object
        mask = (mask > 0).astype(np.uint8)
        if mask.shape != rgb.shape[:2]:
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(mask > 0)
        if len(xs) < max(1, self.min_mask_area):
            return None, None, score
        bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
        return mask, bbox, score

