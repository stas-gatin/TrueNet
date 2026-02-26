import os
import torch
from PIL import Image, ImageFile
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision import transforms
from features import extract_features
import yaml

# ====================== IMAGE SETTINGS ======================
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# ====================== LOAD CONFIG ======================
with open("configs/features_pipeline.yaml", "r") as f:
    cfg = yaml.safe_load(f)

DATA_ROOT = cfg["data_root"]
CACHE_DIR = cfg["cache_dir"]
CHUNK_SIZE = cfg.get("chunk_size", 16)  # Size of a single chunk
MAX_WORKERS = cfg.get("max_workers", 8)
IMAGE_SIZE_THRESHOLD = cfg.get("image_size_threshold", 4000000)
MIN_FILE_SIZE = cfg.get("min_file_size", 5000)
RESIZED_IMAGE_SIZE = tuple(cfg.get("resized_image_size", (1024, 1024)))
INPUT_SIZE = cfg.get("input_size", 224)
FEATURE_CONFIG = cfg.get("feature_config", {"size": 256, "low_cutoff": 0.2, "high_cutoff": 0.6})
MEAN = cfg.get("imagenet_mean", [0.485, 0.456, 0.406])
STD = cfg.get("imagenet_std", [0.229, 0.224, 0.225])
FLIP_PROB = cfg.get("augmentation_flip_prob", 0.5)

# ====================== TRANSFORMS ======================
def pil_to_rgb(img: Image.Image) -> Image.Image:
    """
    Convert a PIL image to an RGB image, adding white background for transparency.
    """
    if img.mode in ("P", "LA", "PA"):
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        return background
    return img

def build_transforms(train: bool = True) -> transforms.Compose:
    """
    Builds the image transformation pipeline.
    
    Args:
        train (bool): Indicates if training splits are created.
        
    Returns:
        Compose structure with sequence of transforms.
    """
    t = [transforms.Lambda(pil_to_rgb),
         transforms.Resize((INPUT_SIZE, INPUT_SIZE))]
    if train and FLIP_PROB > 0:
        t.append(transforms.RandomHorizontalFlip(p=FLIP_PROB))
    t.extend([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
    return transforms.Compose(t)

# ====================== IMAGE PROCESSING ======================
def process_single_image(args: tuple, transforms_comp: transforms.Compose, 
                         image_size_threshold: int, resized_image_size: tuple, 
                         feature_cfg: dict) -> tuple:
    """
    Callback function that processes a single image for multi-threading worker.
    
    Args:
        args (tuple): A tuple containing path and label.
        transforms_comp (Compose): PyTorch transformations logic.
        image_size_threshold (int): Max pixel threshold size for shrinking.
        resized_image_size (tuple): Target dimensions for resizing.
        feature_cfg (dict): Hyperparameters for the feature selection block.
        
    Returns:
        tuple containing path, augmented tensor, feature vectors, and label.
    """
    path, label = args
    try:
        with Image.open(path) as img:
            img = img.convert('RGB')
            if img.size[0] * img.size[1] > image_size_threshold:
                img = img.resize(resized_image_size, Image.Resampling.LANCZOS)

            img_tensor = transforms_comp(img)
            img_np = np.array(img)
            feats = extract_features(img_np, feature_cfg)

            feats = np.array(feats, dtype=np.float32).flatten()
        return path, img_tensor, feats, label

    except Exception as e:
        print(f"Error: {path} \u2192 {e}")
        zero_tensor = torch.zeros(3, INPUT_SIZE, INPUT_SIZE)
        zero_feats = np.zeros(FEATURE_CONFIG.get("size", 256), dtype=np.float32)
        return path, zero_tensor, zero_feats, label

# ====================== FOLDER PROCESSING ======================
def process_folder(real_folder: str, fake_folder: str, cache_subdir: str, train: bool = True) -> None:
    """
    Processes a directory containing real and fake nested images subfolders into a feature pipeline config sequence.
    
    Args:
        real_folder (str): Real image paths.
        fake_folder (str): Fake image paths.
        cache_subdir (str): Path of caching store location.
        train (bool): Determine if logic implies augmentation rules or generic valid testing logic constraints.
    """
    os.makedirs(cache_subdir, exist_ok=True)
    print(f"\nProcessing: {real_folder} + {fake_folder}")

    def get_paths(folder: str) -> list:
        paths = []
        for f in os.listdir(folder):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(folder, f)
                if os.path.getsize(full_path) > MIN_FILE_SIZE:
                    paths.append(full_path)
        return paths

    real_paths = [(p, 0) for p in get_paths(real_folder)]
    fake_paths = [(p, 1) for p in get_paths(fake_folder)]
    all_items = real_paths + fake_paths
    print(f"Found: {len(real_paths)} real, {len(fake_paths)} fake \u2192 {len(all_items)}")

    transforms_comp = build_transforms(train=train)
    chunk_idx = 0
    features_list, labels_list, valid_paths, images_list = [], [], [], []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_image, item, transforms_comp,
                                   IMAGE_SIZE_THRESHOLD, RESIZED_IMAGE_SIZE, FEATURE_CONFIG)
                   for item in all_items]

        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Extracting features")):
            path, img_tensor, feats, label = future.result()
            images_list.append(img_tensor)
            features_list.append(feats)
            labels_list.append(label)
            valid_paths.append(path)

            # Save chunks
            if len(features_list) >= CHUNK_SIZE or i == len(futures) - 1:
                chunk_file = os.path.join(cache_subdir, f"{'train' if train else 'test'}_{chunk_idx:03d}.pt")
                torch.save({
                    'images': torch.stack(images_list),
                    'features': torch.tensor(features_list, dtype=torch.float32),
                    'labels': torch.tensor(labels_list, dtype=torch.long),
                    'paths': valid_paths
                }, chunk_file)
                print(f"Saved chunk {chunk_idx}: {len(features_list)} items \u2192 {chunk_file}")

                # Cleanup for the next chunk
                images_list, features_list, labels_list, valid_paths = [], [], [], []
                chunk_idx += 1

# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    process_folder(
        real_folder=os.path.join(DATA_ROOT, "train", "real"),
        fake_folder=os.path.join(DATA_ROOT, "train", "fake"),
        cache_subdir=os.path.join(CACHE_DIR, "train"),
        train=True
    )

    process_folder(
        real_folder=os.path.join(DATA_ROOT, "test", "real"),
        fake_folder=os.path.join(DATA_ROOT, "test", "fake"),
        cache_subdir=os.path.join(CACHE_DIR, "test"),
        train=False
    )
