# SPDX-License-Identifier: MIT
# Copyright (C) 2025 Rio Rifqi Syah Akbar

import os
import shutil
import random
from math import floor

body_part_folders = ["back-leg", "body", "front-leg", "tail"]
base_path = os.getcwd()

# Set a seed for reproducibility if needed
random.seed(42)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_and_move_images(body_part_path, cat_name):
    cat_path = os.path.join(body_part_path, cat_name)
    if not os.path.exists(cat_path):
        print(f"Cat folder not found: {cat_path}")
        return

    all_images = [f for f in os.listdir(cat_path) if os.path.isfile(os.path.join(cat_path, f))]
    total = len(all_images)

    if total < 3:
        print(f"Not enough images for {cat_name} in {body_part_path} (found {total}, need at least 3)")
        return

    random.shuffle(all_images)

    n_train = floor(total * 0.7)
    n_val = floor(total * 0.15)
    n_test = total - n_train - n_val  # Assign the remainder to test

    train_imgs = all_images[:n_train]
    val_imgs = all_images[n_train:n_train + n_val]
    test_imgs = all_images[n_train + n_val:]

    splits = [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]

    for split_name, split_files in splits:
        dest_dir = os.path.join(body_part_path, split_name, cat_name)
        ensure_dir(dest_dir)

        for file in split_files:
            src = os.path.join(cat_path, file)
            dst = os.path.join(dest_dir, file)
            shutil.move(src, dst)

        print(f"Moved {len(split_files):>2} images of {cat_name} → {split_name}/ in {os.path.basename(body_part_path)}")

def main():
    for body_part in body_part_folders:
        body_part_path = os.path.join(base_path, body_part)

        if not os.path.isdir(body_part_path):
            print(f"Body part folder not found: {body_part_path}")
            continue

        for cat_name in os.listdir(body_part_path):
            cat_path = os.path.join(body_part_path, cat_name)
            if os.path.isdir(cat_path) and cat_name not in ["train", "val", "test"]:
                split_and_move_images(body_part_path, cat_name)

if __name__ == "__main__":
    main()
