"""
Config-driven dataset classes and dataloaders.
"""
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import numpy as np
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class ImageFeatureDataset(Dataset):
    """
    Dataset that loads images and extracts features on-the-fly or from cache.
    
    Args:
        image_paths (list): List of image file paths
        labels (list): List of labels (same length as image_paths)
        feature_extractor (Callable, optional): Function to extract features (image -> features)
        transform (Callable, optional): Optional image transforms
        use_cache (bool): Whether to use cached features
        cache_dir (Path, optional): Directory for cached features
    """
    
    def __init__(
        self,
        image_paths: list,
        labels: list,
        feature_extractor: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        use_cache: bool = False,
        cache_dir: Optional[Path] = None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if len(image_paths) != len(labels):
            raise ValueError(f"Image paths ({len(image_paths)}) and labels ({len(labels)}) must have same length")
    
    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> tuple:
        """Fetch item by index."""
        # Load image
        try:
            with Image.open(self.image_paths[idx]) as img:
                img = img.convert('RGB')
                
                # Resize if too large
                if img.size[0] * img.size[1] > 6_000_000:
                    scale = (6_000_000 / (img.size[0] * img.size[1])) ** 0.5
                    img = img.resize(
                        (int(img.width * scale), int(img.height * scale)),
                        Image.LANCZOS
                    )
                
                # Apply transforms
                if self.transform:
                    image = self.transform(img)
                else:
                    image = transforms.ToTensor()(img)
        except Exception as e:
            # Return zero tensor on error
            image = torch.zeros(3, 224, 224)
        
        # Extract or load features
        if self.feature_extractor:
            # Convert to numpy for feature extraction
            img_np = np.array(img) if isinstance(img, Image.Image) else image.permute(1, 2, 0).numpy()
            features = self.feature_extractor(img_np, {})  # Pass empty config if not needed
        else:
            features = torch.tensor([], dtype=torch.float32)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, features, label


class CachedFeatureDataset(Dataset):
    """
    Dataset that loads pre-computed features from cache.
    Supports two cache schemes:
      1) legacy: {'features', 'labels', 'paths'}  -> opens images on the fly
      2) tensors: {'images', 'features', 'labels', 'paths'} -> returns tensors directly
      
    Args:
        cache_file (Path): Path to the cache configuration file.
        transform (Callable, optional): Transformations to apply to the dataset.
    """
    def __init__(self, cache_file: Path, transform: Optional[Callable] = None):
        print(f"Loading cache: {cache_file}")
        data = torch.load(cache_file, map_location="cpu")
        # determine mode by keys present
        self.has_images_tensor = 'images' in data
        self.transform = transform

        if self.has_images_tensor:
            # New cached-tensors format
            self.images = data['images']            # Tensor[N, 3, H, W]
            self.features = data['features']        # Tensor[N, F]
            self.labels = data['labels']            # Tensor[N]
            self.paths = data.get('paths', [None] * len(self.labels))
            # Basic sanity checks
            assert len(self.images) == len(self.features) == len(self.labels), \
                "Cached tensors 'images','features','labels' must have same length"
            print(f"Dataset (tensor cache): {len(self)} items (images tensor present)")
        else:
            # Legacy format: features + paths
            self.features = data['features']
            self.labels = data['labels']
            self.paths = data['paths']
            print(f"Dataset (legacy cache): {len(self)} items (images will be opened on the fly)")

        # convert everything to tensors if necessary
        if not isinstance(self.features, torch.Tensor):
            self.features = torch.tensor(self.features, dtype=torch.float32)
        if not isinstance(self.labels, torch.Tensor):
            self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        """Get dataset item by index."""
        if self.has_images_tensor:
            img = self.images[idx]
            feat = self.features[idx].clone()
            label = self.labels[idx].clone()
            # If transform exists, apply it only if img is not a tensor
            if self.transform and not isinstance(img, torch.Tensor):
                img = self.transform(img)
            return img, feat, label

        # legacy: open image and apply transform (backward compatible)
        path = self.paths[idx]
        try:
            with Image.open(path) as img:
                img = img.convert('RGB')
                # optional resize guard (kept minimal)
                if self.transform:
                    image = self.transform(img)
                else:
                    image = transforms.ToTensor()(img)
        except Exception:
            image = torch.zeros(3, 224, 224)

        feat = self.features[idx].clone()
        label = self.labels[idx].clone()
        return image, feat, label


class ChunkedFeatureDataset(IterableDataset):
    """
    Dataset that lazily loads pre-computed chunks (.pt) 
    containing images and features. Supports sequential
    fetching and shuffling within a chunk and/or across chunks.
    
    Args:
        cache_dir (str): Cache directory path.
        transform (Callable, optional): Transforms to apply.
        shuffle_chunks (bool): Whether to shuffle the chunks prior to iteration computation.
        shuffle_within_chunk (bool): Whether to shuffle images inside a chunk.
    """

    def __init__(self, cache_dir: str, transform=None, shuffle_chunks=True, shuffle_within_chunk=True):
        self.cache_dir = Path(cache_dir)
        self.transform = transform
        self.chunk_files = sorted(self.cache_dir.glob("*.pt"))
        if not self.chunk_files:
            raise FileNotFoundError(f"No .pt files found in {cache_dir}")

        self.shuffle_chunks = shuffle_chunks
        self.shuffle_within_chunk = shuffle_within_chunk

        # Calculate total number of items
        self.total_len = sum(len(torch.load(f, map_location="cpu")['labels']) for f in self.chunk_files)


    def __iter__(self):
        """Iterator for streaming records."""
        # Determine chunk order
        chunk_order = list(range(len(self.chunk_files)))
        if self.shuffle_chunks:
            chunk_order = torch.randperm(len(self.chunk_files)).tolist()

        for idx in chunk_order:
            # Lazily load one chunk
            chunk = torch.load(self.chunk_files[idx], map_location="cpu")
            images, features, labels = chunk['images'], chunk['features'], chunk['labels']

            # Determine order of elements inside the chunk
            indices = list(range(len(labels)))
            if self.shuffle_within_chunk:
                indices = torch.randperm(len(labels)).tolist()

            for i in indices:
                img, feat, label = images[i], features[i], labels[i]
                if self.transform:
                    img = self.transform(img)
                yield img, feat, label

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return self.total_len
    

class ChunkedFeatureDatasetOLD(torch.utils.data.IterableDataset):
    """
    Old Chunked Feature Dataset with multithreading support.
    
    Args:
        cache_dir (str): Cache directory path.
        transform (Callable, optional): Transforms to apply.
        shuffle_chunks (bool): Whether to shuffle the chunks prior to iteration computation.
        shuffle_within_chunk (bool): Whether to shuffle images inside a chunk.
    """
    def __init__(self, cache_dir: str, transform=None, shuffle_chunks=True, shuffle_within_chunk=True):
        self.cache_dir = Path(cache_dir)
        self.transform = transform
        self.chunk_files = sorted(self.cache_dir.glob("*.pt"))
        if not self.chunk_files:
            raise FileNotFoundError(f"No .pt files found in {cache_dir}")

        self.shuffle_chunks = shuffle_chunks
        self.shuffle_within_chunk = shuffle_within_chunk
        self.total_len = sum(len(torch.load(f, map_location="cpu")['labels']) for f in self.chunk_files)

    def __iter__(self):
        """Streaming iterates via dataloader worker shards."""
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process
            iter_chunk_files = self.chunk_files
        else:
            # Multi-process: split chunk_files among workers
            per_worker = int(np.ceil(len(self.chunk_files) / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.chunk_files))
            iter_chunk_files = self.chunk_files[start:end]

        chunk_order = list(range(len(iter_chunk_files)))
        if self.shuffle_chunks:
            chunk_order = torch.randperm(len(iter_chunk_files)).tolist()

        for idx in chunk_order:
            chunk = torch.load(iter_chunk_files[idx], map_location="cpu")
            images, features, labels = chunk['images'], chunk['features'], chunk['labels']

            indices = list(range(len(labels)))
            if self.shuffle_within_chunk:
                indices = torch.randperm(len(labels)).tolist()

            for i in indices:
                img, feat, label = images[i], features[i], labels[i]
                if self.transform and not isinstance(img, torch.Tensor):
                    img = self.transform(img)
                yield img, feat, label

    def __len__(self) -> int:
        """Returns dataset length."""
        return self.total_len


def get_transforms(cfg: Dict[str, Any], is_train: bool = True) -> transforms.Compose:
    """
    Create transforms from config.
    
    Args:
        cfg: Config dictionary with 'data' key
        is_train: Whether to use training transforms (with augmentation)
        
    Returns:
        Compose transform
    """
    img_size = cfg['data'].get('img_size', 256)
    
    if is_train:
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    else:
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    
    return transforms.Compose(transform_list)


def create_dataloader(
    dataset: Dataset,
    cfg: Dict[str, Any],
    shuffle: bool = True
) -> DataLoader:
    """
    Create DataLoader from dataset and config.
    
    Args:
        dataset: PyTorch dataset
        cfg: Config dictionary with 'data' key
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=shuffle,
        num_workers=cfg['data'].get('num_workers', 4),
        pin_memory=cfg['data'].get('pin_memory', False),
        persistent_workers=cfg['data'].get('num_workers', 0) > 0
    )
