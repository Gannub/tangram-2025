import os
import random
import json
from PIL import Image, ImageDraw
from svgpathtools import svg2paths
import re
import numpy as np
from detectron2.structures import BoxMode

# Paths to SVG files
svg_files = [
    'mnt/data/Large Triangle 1.svg',
    'mnt/data/Large Triangle 2.svg',
    'mnt/data/Medium Triangle.svg',
    'mnt/data/Small Triangle 1.svg',
    'mnt/data/Small Triangle 2.svg',
    'mnt/data/Square.svg',
    'mnt/data/Parallelogram.svg'
]

# Canvas size
CANVAS_SIZE = 1024
# Scale factor for resizing SVG shapes
SCALE_FACTOR = 0.05

# COCO category mapping
categories = [
    {"id": 1, "name": "Large Triangle 1"},
    {"id": 2, "name": "Large Triangle 2"},
    {"id": 3, "name": "Medium Triangle"},
    {"id": 4, "name": "Small Triangle 1"},
    {"id": 5, "name": "Small Triangle 2"},
    {"id": 6, "name": "Square"},
    {"id": 7, "name": "Parallelogram"}
]

def load_svg(svg_path):
    paths, attributes = svg2paths(svg_path)
    return paths, attributes

def extract_fill_color(attributes):
    """Extracts fill color from the 'style' attribute if available."""
    if attributes and 'style' in attributes[0]:
        style = attributes[0]['style']
        match = re.search(r'fill: ?(#\w+);?', style)
        if match:
            return match.group(1)
    return '#000000'  # Default to black if no color is found

def get_bounding_box(shape):
    """Calculates the bounding box of a shape."""
    x_coords = [point[0] for segment in shape for point in segment]
    y_coords = [point[1] for segment in shape for point in segment]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

def clip_to_canvas(points, canvas_size):
    """Clips the polygon points to the canvas dimensions."""
    clipped_points = []
    for x, y in points:
        x_clipped = min(max(x, 0), canvas_size)  # Clip x within [0, canvas_size]
        y_clipped = min(max(y, 0), canvas_size)  # Clip y within [0, canvas_size]
        clipped_points.append((x_clipped, y_clipped))
    return clipped_points

def is_valid_segmentation(segmentation, canvas_size):
    """Checks if a segmentation has valid points within the canvas."""
    x_coords = segmentation[0::2]
    y_coords = segmentation[1::2]
    return any(0 <= x < canvas_size and 0 <= y < canvas_size for x, y in zip(x_coords, y_coords))

# Generate COCO-style dataset
def generate_coco_dataset(svg_files, num_samples=1000, output_dir="dataset"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)  # Ensure images directory exists

    images = []
    annotations = []
    annotation_id = 1

    for sample_id in range(num_samples):
        # Create blank canvas
        canvas = Image.new('RGBA', (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        image_info = {
            "file_name": f"images/sample_{sample_id}.png",
            "id": sample_id,
            "height": CANVAS_SIZE,
            "width": CANVAS_SIZE
        }
        images.append(image_info)

        placement_mode = random.choices(
            ['all_shapes', 'random_shapes', 'single_shape'],
            weights=[0.5, 0.3, 0.2],
            k=1
        )[0]

        if placement_mode == 'all_shapes':
            selected_shapes = svg_files
        elif placement_mode == 'random_shapes':
            selected_shapes = random.sample(svg_files, random.randint(1, len(svg_files)))
        elif placement_mode == 'single_shape':
            selected_shapes = [random.choice(svg_files)]

        for svg_file in selected_shapes:
            # Load SVG
            paths, attributes = load_svg(svg_file)

            # Extract fill color
            fill_color = extract_fill_color(attributes)

            # Apply scaling and calculate transformed shape
            transformed_shape = []
            for path in paths:
                for segment in path:
                    start = segment.start * SCALE_FACTOR
                    end = segment.end * SCALE_FACTOR
                    transformed_shape.append(((start.real, start.imag), (end.real, end.imag)))

            # Calculate bounding box and ensure shape fits within canvas
            min_x, min_y, max_x, max_y = get_bounding_box(transformed_shape)
            width, height = max_x - min_x, max_y - min_y

            max_offset_x = CANVAS_SIZE - max_x
            max_offset_y = CANVAS_SIZE - max_y
            x_offset = random.randint(-int(min_x), int(max_offset_x))
            y_offset = random.randint(-int(min_y), int(max_offset_y))
            rotation = random.choice([0, 90, 180, 270])

            # Apply rotation and offset
            final_shape = []
            for segment in transformed_shape:
                rotated_segment = []
                for point in segment:
                    angle = np.deg2rad(rotation)
                    x = np.cos(angle) * point[0] - np.sin(angle) * point[1]
                    y = np.sin(angle) * point[0] + np.cos(angle) * point[1]
                    x += x_offset
                    y += y_offset
                    rotated_segment.append((x, y))
                final_shape.append(rotated_segment)

            # Verify the shape stays within bounds
            all_x = [point[0] for segment in final_shape for point in segment]
            all_y = [point[1] for segment in final_shape for point in segment]
            if min(all_x) < 0 or max(all_x) > CANVAS_SIZE or min(all_y) < 0 or max(all_y) > CANVAS_SIZE:
                continue  # Skip this shape if it goes out of bounds

            # Flatten final_shape for segmentation
            segmentation = [coord for segment in final_shape for point in segment for coord in point]
            
            # Update bounding box based on transformed shape
            bbox_min_x = min(segmentation[0::2])
            bbox_min_y = min(segmentation[1::2])
            bbox_max_x = max(segmentation[0::2])
            bbox_max_y = max(segmentation[1::2])
            bbox_width = bbox_max_x - bbox_min_x
            bbox_height = bbox_max_y - bbox_min_y

            # Draw the filled polygon on the canvas
            polygon = [point for segment in final_shape for point in segment]
            draw.polygon(polygon, fill=fill_color)

            # Save annotation
            annotation = {
                "id": annotation_id,
                "image_id": sample_id,
                "category_id": next(c["id"] for c in categories if c["name"] in svg_file),
                "bbox": [bbox_min_x, bbox_min_y, bbox_width, bbox_height],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [segmentation],  # Flattened polygon points
                "area": width * height,
                "iscrowd": 0
            }
            annotations.append(annotation)
            annotation_id += 1

        # Save image
        canvas.save(f"{output_dir}/images/sample_{sample_id}.png")

    # Split dataset into train/test
    train_ratio = 0.8
    num_train = int(len(images) * train_ratio)
    train_images = images[:num_train]
    test_images = images[num_train:]

    train_annotations = [ann for ann in annotations if ann["image_id"] < num_train]
    test_annotations = [ann for ann in annotations if ann["image_id"] >= num_train]

    train_output = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories
    }
    test_output = {
        "images": test_images,
        "annotations": test_annotations,
        "categories": categories
    }

    # Save COCO JSON
    with open(f"{output_dir}/train.json", "w") as f:
        json.dump(train_output, f)
    with open(f"{output_dir}/test.json", "w") as f:
        json.dump(test_output, f)

if __name__ == "__main__":
    generate_coco_dataset(svg_files, num_samples=100)
