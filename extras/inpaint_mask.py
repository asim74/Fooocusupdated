import sys

import modules.config
import numpy as np
import torch
from extras.GroundingDINO.util.inference import default_groundingdino
from extras.sam.predictor import SamPredictor
from rembg import remove, new_session
from segment_anything import sam_model_registry
from segment_anything.utils.amg import remove_small_regions


class SAMOptions:
    def __init__(self,
                 # GroundingDINO
                 dino_prompt: str = '',
                 dino_box_threshold=0.3,
                 dino_text_threshold=0.25,
                 dino_erode_or_dilate=0,
                 dino_debug=False,

                 # SAM
                 max_detections=2,
                 model_type='vit_b',
                 
                 # NEW: Add Point coordinates for click-to-select
                 point_coords=None,
                 point_labels=None
                 ):
        self.dino_prompt = dino_prompt
        self.dino_box_threshold = dino_box_threshold
        self.dino_text_threshold = dino_text_threshold
        self.dino_erode_or_dilate = dino_erode_or_dilate
        self.dino_debug = dino_debug
        self.max_detections = max_detections
        self.model_type = model_type
        self.point_coords = point_coords
        self.point_labels = point_labels


def optimize_masks(masks: torch.Tensor) -> torch.Tensor:
    """
    removes small disconnected regions and holes
    """
    fine_masks = []
    for mask in masks.to('cpu').numpy():  # masks: [num_masks, 1, h, w]
        fine_masks.append(remove_small_regions(mask[0], 400, mode="holes")[0])
    masks = np.stack(fine_masks, axis=0)[:, np.newaxis]
    return torch.from_numpy(masks)


def generate_mask_from_image(image: np.ndarray, mask_model: str = 'sam', extras=None,
                             sam_options: SAMOptions | None = SAMOptions) -> tuple[np.ndarray | None, int | None, int | None, int | None]:
    dino_detection_count = 0
    sam_detection_count = 0
    sam_detection_on_mask_count = 0

    if image is None:
        return None, dino_detection_count, sam_detection_count, sam_detection_on_mask_count

    if extras is None:
        extras = {}

    if 'image' in image:
        image = image['image']

    if mask_model != 'sam' or sam_options is None:
        result = remove(
            image,
            session=new_session(mask_model, **extras),
            only_mask=True,
            **extras
        )

        return result, dino_detection_count, sam_detection_count, sam_detection_on_mask_count

    # Load SAM Model
    sam_checkpoint = modules.config.download_sam_model(sam_options.model_type)
    sam = sam_model_registry[sam_options.model_type](checkpoint=sam_checkpoint)

    sam_predictor = SamPredictor(sam)
    final_mask_tensor = torch.zeros((image.shape[0], image.shape[1]))
    
    sam_predictor.set_image(image)

    # --- NEW LOGIC: 1. Check for Point-Based Click Selection First ---
    if sam_options.point_coords is not None and len(sam_options.point_coords) > 0:
        point_coords = np.array(sam_options.point_coords)
        point_labels = np.array(sam_options.point_labels)
        
        # Transform coords for SAM
        transformed_coords = sam_predictor.transform.apply_coords(point_coords, image.shape[:2])
        coords_torch = torch.as_tensor(transformed_coords, dtype=torch.float, device=sam_predictor.device)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=sam_predictor.device)
        
        # Predict_torch expects BxNx2 for coords and BxN for labels, add batch dimension
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

        masks, _, _ = sam_predictor.predict_torch(
            point_coords=coords_torch,
            point_labels=labels_torch,
            boxes=None,
            multimask_output=False,
        )

        masks = optimize_masks(masks)
        sam_detection_count = len(masks)
        if sam_detection_count > 0:
            final_mask_tensor += masks[0][0]
            sam_detection_on_mask_count += 1

    # --- EXISTING LOGIC: 2. Fallback to Text-Based GroundingDINO Selection ---
    elif sam_options.dino_prompt and sam_options.dino_prompt.strip():
        detections, boxes, logits, phrases = default_groundingdino(
            image=image,
            caption=sam_options.dino_prompt,
            box_threshold=sam_options.dino_box_threshold,
            text_threshold=sam_options.dino_text_threshold
        )

        H, W = image.shape[0], image.shape[1]
        boxes = boxes * torch.Tensor([W, H, W, H])
        boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
        boxes[:, 2:] = boxes[:, 2:] + boxes[:, :2]

        dino_detection_count = boxes.size(0)

        if dino_detection_count > 0:
            if sam_options.dino_erode_or_dilate != 0:
                for index in range(boxes.size(0)):
                    assert boxes.size(1) == 4
                    boxes[index][0] -= sam_options.dino_erode_or_dilate
                    boxes[index][1] -= sam_options.dino_erode_or_dilate
                    boxes[index][2] += sam_options.dino_erode_or_dilate
                    boxes[index][3] += sam_options.dino_erode_or_dilate

            if sam_options.dino_debug:
                from PIL import ImageDraw, Image
                debug_dino_image = Image.new("RGB", (image.shape[1], image.shape[0]), color="black")
                draw = ImageDraw.Draw(debug_dino_image)
                for box in boxes.numpy():
                    draw.rectangle(box.tolist(), fill="white")
                return np.array(debug_dino_image), dino_detection_count, sam_detection_count, sam_detection_on_mask_count

            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            masks = optimize_masks(masks)
            sam_detection_count = len(masks)
            if sam_options.max_detections == 0:
                sam_options.max_detections = sys.maxsize
            sam_objects = min(len(logits), sam_options.max_detections)
            for obj_ind in range(sam_objects):
                mask_tensor = masks[obj_ind][0]
                final_mask_tensor += mask_tensor
                sam_detection_on_mask_count += 1

    # Convert the processed mask back to an image
    final_mask_tensor = (final_mask_tensor > 0).to('cpu').numpy()
    mask_image = np.dstack((final_mask_tensor, final_mask_tensor, final_mask_tensor)) * 255
    mask_image = np.array(mask_image, dtype=np.uint8)
    return mask_image, dino_detection_count, sam_detection_count, sam_detection_on_mask_count
