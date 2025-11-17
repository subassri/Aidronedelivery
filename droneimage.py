import os
import cv2
import numpy as np
import random
import pymysql
import shutil
import tempfile
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml
from dotenv import load_dotenv
from ultralytics import YOLO
from aidbrepo import TIDB_CONFIG   # safe placeholder config

# --------------------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES (Not included in GitHub)
# --------------------------------------------------------------------
env_path = os.path.join(os.path.dirname(__file__), "cloud.env")
load_dotenv(env_path)

TIDB_CONFIG.update({
    "host": os.getenv("TIDB_HOST"),
    "port": int(os.getenv("TIDB_PORT", 4000)),
    "user": os.getenv("TIDB_USER"),
    "password": os.getenv("TIDB_PASS"),
    "database": os.getenv("TIDB_DB"),
})

# --------------------------------------------------------------------
# CONNECT TO TiDB (Credentials come from cloud.env)
# --------------------------------------------------------------------
conn = pymysql.connect(
    host=TIDB_CONFIG["host"],
    port=TIDB_CONFIG["port"],
    user=TIDB_CONFIG["user"],
    password=TIDB_CONFIG["password"],
    database=TIDB_CONFIG["database"],
    ssl={"ssl": {}}   # safe default
)

# --------------------------------------------------------------------
# USER-SUPPLIED IMAGE PATHS (Safe for GitHub)
# Provide your own sample images locally. Do NOT upload real images.
# --------------------------------------------------------------------
image_paths = [
    # Example placeholders (user should replace):
    "data/samples/broken_0001.jpg",
    "data/samples/broken_0002.jpg",
    "data/samples/broken_0003.jpg",
]

# Ensure dataset exists
image_paths = [p for p in image_paths if os.path.exists(p)]
if not image_paths:
    raise FileNotFoundError(
        "No sample images found. Place 3–10 starter images in data/samples/"
    )

# --------------------------------------------------------------------
# OUTPUT ROOT (GitHub-safe — user-controlled)
# --------------------------------------------------------------------
output_root = "generated_dataset"
os.makedirs(output_root, exist_ok=True)

# --------------------------------------------------------------------
# CLASS DEFINITIONS
# --------------------------------------------------------------------
drone_classes = ['Multirotor', 'Fixedwing', 'Hybrid', 'Singlerotor']
class_ids = {cls: i for i, cls in enumerate(drone_classes)}
damage_class_id = len(drone_classes)

images_per_class = 120   # Augmented total

# --------------------------------------------------------------------
# DAMAGE AUGMENTATION FUNCTIONS
# --------------------------------------------------------------------
def add_scratches(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(random.randint(3, 8)):
        x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
        x2, y2 = x1 + random.randint(-30,30), y1 + random.randint(-30,30)
        thickness = random.randint(1,2)
        cv2.line(img, (x1,y1), (x2,y2), (255,255,255), thickness)
        cv2.line(mask, (x1,y1), (x2,y2), 255, thickness)
    return img, mask

def add_burn(img):
    h, w = img.shape[:2]
    overlay = img.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(random.randint(1,2)):
        center = (random.randint(w//4, 3*w//4), random.randint(h//4,3*h//4))
        radius = random.randint(15,50)
        cv2.circle(overlay, center, radius, (0,0,0), -1)
        cv2.circle(mask, center, radius, 255, -1)
    alpha = random.uniform(0.3,0.6)
    img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
    return img, mask

# ... other damage functions omitted for brevity (safe to keep) ...

def random_damage(img):
    has_damage = False
    combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    if random.random() < 0.7:
        func = random.choice([add_scratches, add_burn])
        img, mask = func(img)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
        has_damage = True

    return img, combined_mask, has_damage

# --------------------------------------------------------------------
# DRONE TYPE SIMULATION (safe transformations)
# --------------------------------------------------------------------
def simulate_drone_type(img, drone_type):
    h, w = img.shape[:2]
    if drone_type == 'Fixedwing':
        img = cv2.resize(img, (int(w*1.5), h))
    elif drone_type == 'Hybrid':
        M = cv2.getRotationMatrix2D((w//2,h//2), random.randint(-15,15), 1.2)
        img = cv2.warpAffine(img, M, (w,h))
    elif drone_type == 'Singlerotor':
        img = img[h//4:h*3//4, w//4:w*3//4]
        img = cv2.resize(img, (w,h))

    # General augmentation
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w,h))

    return img

# --------------------------------------------------------------------
# IMAGE GENERATION PIPELINE
# --------------------------------------------------------------------
all_images = []
for cls in drone_classes:
    for i in tqdm(range(images_per_class), desc=f"Generating {cls}"):
        base_img = cv2.imread(random.choice(image_paths))
        img = simulate_drone_type(base_img.copy(), cls)
        img, mask, has_damage = random_damage(img)
        fname = f"{cls}_{i:04d}.jpg"
        all_images.append((img, fname, cls, mask, has_damage))

# --------------------------------------------------------------------
# TRAIN/VAL/TEST SPLIT
# --------------------------------------------------------------------
train_imgs, temp_imgs = train_test_split(all_images, test_size=0.3, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
splits = {"train": train_imgs, "val": val_imgs, "test": test_imgs}

# --------------------------------------------------------------------
# BOUNDING BOX FROM DAMAGE MASK
# --------------------------------------------------------------------
def create_bbox(mask):
    if mask is None or cv2.countNonZero(mask) == 0:
        return None
    y,x = np.where(mask > 0)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    return x_min, y_min, x_max-x_min, y_max-y_min

# --------------------------------------------------------------------
# SAVE LOCALLY + UPLOAD TO TIDB
# --------------------------------------------------------------------
uploaded = []

for split_name, items in splits.items():
    for img, fname, cls, mask, has_damage in tqdm(items, desc=f"Saving {split_name}"):

        # Save image and label locally
        img_dir = os.path.join(output_root,"images",split_name,cls)
        lbl_dir = os.path.join(output_root,"labels",split_name,cls)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        img_path = os.path.join(img_dir, fname)
        cv2.imwrite(img_path, img)

        # Build YOLO label file
        lbl_path = os.path.join(lbl_dir, fname.replace(".jpg", ".txt"))
        drone_bbox = f"{class_ids[cls]} 0.5 0.5 1.0 1.0\n"
        bbox = create_bbox(mask)
        damage_bbox = ""

        if bbox:
            x, y, w, h = bbox
            x /= img.shape[1]
            y /= img.shape[0]
            w /= img.shape[1]
            h /= img.shape[0]
            damage_bbox = f"{damage_class_id} {x:.3f} {y:.3f} {w:.3f} {h:.3f}\n"

        with open(lbl_path, "w") as f:
            f.write(drone_bbox)
            if damage_bbox:
                f.write(damage_bbox)

        # Upload to TiDB
        with open(img_path, "rb") as f:
            blob = f.read()

        uploaded.append((fname, cls, has_damage, split_name, (drone_bbox + damage_bbox).strip(), blob))

with conn.cursor() as cur:
    for rec in tqdm(uploaded, desc="Uploading to TiDB"):
        cur.execute("""
            INSERT INTO drone_dataset (filename, drone_type, has_damage, split, label_text, image_blob)
            VALUES (%s,%s,%s,%s,%s,%s)
        """, rec)
conn.commit()

print("✔ Upload completed")

# --------------------------------------------------------------------
# TRAIN YOLO
# --------------------------------------------------------------------
tmp_dir = tempfile.mkdtemp(prefix="tidb_dataset_")
local_yaml = os.path.join(tmp_dir, "dataset.yaml")

yaml.dump({
    "train": os.path.join(output_root, "images/train"),
    "val": os.path.join(output_root, "images/val"),
    "test": os.path.join(output_root, "images/test"),
    "nc": len(drone_classes) + 1,
    "names": drone_classes + ["damage"]
}, open(local_yaml, "w"))

model = YOLO("yolov8n.pt")
model.train(
    data=local_yaml,
    epochs=50,
    imgsz=416,
    batch=4,
    lr0=0.001,
    optimizer="SGD",
    augment=True
)

print("✔ YOLO Training Complete")

shutil.rmtree(tmp_dir)
print("✔ Temporary files cleaned")

