import os
import jsons
from PIL import Image

BASE_IMAGE_DIR = "image"
OUTPUT_DIR = "dataset_patches"
SPLITS = ["train", "valid", "test"]

total_occupied = 0
total_vacant = 0

print("COCO Dataset processing starting...\n")

for split in SPLITS:
    split_dir = os.path.join(BASE_IMAGE_DIR, split)
    json_path = os.path.join(split_dir, "_annotations.coco.json")

    if not os.path.exists(json_path):
        print(f"Skipping {split}: JSON not found ({json_path})")
        continue

    print(f"--- Processing {split.upper()} folder ---")

    with open(json_path, "r") as f:
        coco_data = json.load(f)

    img_map = {img["id"]: img.get("file_name", "") for img in coco_data.get("images", [])}

    occ_folder = os.path.join(OUTPUT_DIR, split, "occupied")
    vac_folder = os.path.join(OUTPUT_DIR, split, "vacant")
    os.makedirs(occ_folder, exist_ok=True)
    os.makedirs(vac_folder, exist_ok=True)

    split_occ = 0
    split_vac = 0
    annotations = coco_data.get("annotations", [])
    print(f"Total annotations in {split}: {len(annotations)}")

    for ann_idx, ann in enumerate(annotations):
        img_id = ann.get("image_id")
        cat_id = ann.get("category_id")
        bbox = ann.get("bbox")

        if not bbox or img_id not in img_map:
            continue

        # Direct ID check: 2 = Occupied, 1 = Vacant
        if cat_id == 2:
            is_occupied = True
        elif cat_id == 1:
            is_occupied = False
        else:
            continue

        raw_img_name = img_map[img_id]
        clean_img_name = os.path.basename(raw_img_name)

        possible_paths = [
            os.path.join(split_dir, clean_img_name),
            os.path.join(split_dir, raw_img_name),
            os.path.join(BASE_IMAGE_DIR, clean_img_name),
            os.path.join(BASE_IMAGE_DIR, raw_img_name)
        ]

        img_full_path = None
        for p in possible_paths:
            if os.path.exists(p):
                img_full_path = p
                break

        if not img_full_path:
            continue

        try:
            with Image.open(img_full_path) as img:
                x, y, w, h = bbox
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(img.width, int(x + w))
                y2 = min(img.height, int(y + h))

                if (x2 - x1) <= 5 or (y2 - y1) <= 5:
                    continue

                patch = img.crop((x1, y1, x2, y2))
                patch = patch.resize((128, 128))

                save_name = f"{img_id}_{ann_idx}.jpg"
                if is_occupied:
                    patch.save(os.path.join(occ_folder, save_name))
                    split_occ += 1
                else:
                    patch.save(os.path.join(vac_folder, save_name))
                    split_vac += 1
        except Exception:
            continue

        if (ann_idx + 1) % 1000 == 0:
            print(f" {ann_idx + 1} spots processed | Occupied: {split_occ} | Vacant: {split_vac}")

    print(f"{split.upper()} complete! Occupied: {split_occ}, Vacant: {split_vac}\n")
    total_occupied += split_occ
    total_vacant += split_vac

print("====================================")
print("All Patches Saved successfully!")
print(f"Total Occupied Patches: {total_occupied}")
print(f"Total Vacant Patches: {total_vacant}")
print(f"Folder check karein: {OUTPUT_DIR}/")
print("====================================")