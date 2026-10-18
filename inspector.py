import os
import pandas as pd
from PIL import Image
import json
from collections import defaultdict

def analyze_dataset():
    project_root = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(project_root, "Images")

    # 1. GTSRB
    print("=== GTSRB ===")
    gtsrb_dir = os.path.join(images_dir, "gtsrb-german-traffic-sign", "versions", "1")
    
    # annotations
    train_csv = pd.read_csv(os.path.join(gtsrb_dir, "Train.csv"))
    test_csv = pd.read_csv(os.path.join(gtsrb_dir, "Test.csv"))
    meta_csv = pd.read_csv(os.path.join(gtsrb_dir, "Meta.csv"))
    
    print(f"Train.csv: {train_csv.shape}, cols: {list(train_csv.columns)}")
    print(f"Test.csv: {test_csv.shape}, cols: {list(test_csv.columns)}")
    print(f"Meta.csv: {meta_csv.shape}, cols: {list(meta_csv.columns)}")
    
    # image formats & channels
    formats = set()
    channels = set()
    widths = set()
    heights = set()
    corrupted = 0
    missing = 0
    
    train_counts = train_csv['ClassId'].value_counts().to_dict()
    test_counts = test_csv['ClassId'].value_counts().to_dict()
    
    for idx, row in train_csv.head(100).iterrows():
        img_path = os.path.join(gtsrb_dir, row['Path'])
        if not os.path.exists(img_path):
            missing += 1
            continue
        try:
            with Image.open(img_path) as img:
                formats.add(img.format)
                channels.add(img.mode)
                widths.add(img.width)
                heights.add(img.height)
        except Exception:
            corrupted += 1
            
    print(f"Image Formats: {formats}")
    print(f"Channels: {channels}")
    print(f"Dimensions: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
    print(f"Total train classes: {len(train_counts)}")
    print(f"Total test classes: {len(test_counts)}")
    print(f"Train counts: {train_counts}")
    print(f"Test counts: {test_counts}")
    
    # 2. CTSD
    print("\n=== CTSD ===")
    ctsd_dir = os.path.join(images_dir, "chinese-traffic-signs", "versions", "2")
    annotations = pd.read_csv(os.path.join(ctsd_dir, "annotations.csv"))
    print(f"annotations.csv: {annotations.shape}, cols: {list(annotations.columns)}")
    
    formats = set()
    channels = set()
    widths = set()
    heights = set()
    missing = 0
    corrupted = 0
    
    class_counts = annotations['category'].value_counts().to_dict()
    for idx, row in annotations.head(100).iterrows():
        img_path = os.path.join(ctsd_dir, "images", row['file_name'])
        if not os.path.exists(img_path):
            missing += 1
            continue
        try:
            with Image.open(img_path) as img:
                formats.add(img.format)
                channels.add(img.mode)
                widths.add(img.width)
                heights.add(img.height)
        except Exception:
            corrupted += 1
            
    print(f"Image Formats: {formats}")
    print(f"Channels: {channels}")
    print(f"Dimensions: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
    print(f"Total classes: {len(class_counts)}")
    print(f"Class counts: {class_counts}")
    print(f"Missing (sample): {missing}")
    
    # 3. BelgiumTSC
    print("\n=== BelgiumTSC ===")
    belgium_dir = os.path.join(images_dir, "belgium-ts", "versions", "10")
    
    for d in os.listdir(belgium_dir):
        print(f"Subdir: {d}")
        
    train_dir = os.path.join(belgium_dir, "BelgiumTSC_Training", "Training")
    test_dir = os.path.join(belgium_dir, "BelgiumTSC_Testing", "Testing")
    
    train_counts = {}
    test_counts = {}
    
    formats = set()
    channels = set()
    widths = set()
    heights = set()
    
    if os.path.exists(train_dir):
        for c in os.listdir(train_dir):
            if os.path.isdir(os.path.join(train_dir, c)):
                imgs = os.listdir(os.path.join(train_dir, c))
                train_counts[c] = len(imgs)
                if len(imgs) > 0:
                    try:
                        with Image.open(os.path.join(train_dir, c, imgs[0])) as img:
                            formats.add(img.format)
                            channels.add(img.mode)
                            widths.add(img.width)
                            heights.add(img.height)
                    except Exception:
                        pass
                        
    if os.path.exists(test_dir):
        for c in os.listdir(test_dir):
            if os.path.isdir(os.path.join(test_dir, c)):
                test_counts[c] = len(os.listdir(os.path.join(test_dir, c)))

    print(f"Image Formats: {formats}")
    print(f"Channels: {channels}")
    if widths:
        print(f"Dimensions: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
    print(f"Total train classes: {len(train_counts)}")
    print(f"Total test classes: {len(test_counts)}")
    print(f"Train counts: {train_counts}")
    print(f"Test counts: {test_counts}")
    
if __name__ == '__main__':
    analyze_dataset()
