"""
Config-driven feature extraction module.
Supports modular feature extractors that can be enabled/disabled via config.
"""
import numpy as np
import torch
from scipy.fftpack import dct
from skimage import color
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from typing import Dict, Any, Union, List
import os
from pathlib import Path


def fft_features(img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Extract FFT-based frequency features from an image.
    
    Args:
        img: Input image array (RGB or grayscale)
        cfg: Config dict with 'size', 'low_cutoff', 'high_cutoff'
        
    Returns:
        Array of FFT features (3 values: low, mid, high frequency energy)
    """
    size = cfg.get('size', 256)
    low_cutoff = cfg.get('low_cutoff', 0.2)
    high_cutoff = cfg.get('high_cutoff', 0.6)
    
    # Convert to grayscale and resize
    if len(img.shape) == 3:
        gray = color.rgb2gray(img)
    else:
        gray = img
    gray = resize(gray, (size, size), anti_aliasing=True)
    
    # Compute 2D FFT
    F = np.fft.fft2(gray)
    Fshift = np.fft.fftshift(F)
    magnitude = np.abs(Fshift)
    
    # Calculate total energy
    total_energy = np.sum(magnitude**2) + 1e-8
    
    # Create radial coordinate grid
    h, w = gray.shape
    y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
    r = np.sqrt(x**2 + y**2)
    r_max = np.max(r)
    
    # Define frequency bands
    low = r < low_cutoff * r_max
    mid = (r >= low_cutoff * r_max) & (r < high_cutoff * r_max)
    high = r >= high_cutoff * r_max
    
    # Calculate normalized energy
    lowE = np.sum(magnitude[low]**2) / total_energy
    midE = np.sum(magnitude[mid]**2) / total_energy
    highE = np.sum(magnitude[high]**2) / total_energy
    
    return np.array([lowE, midE, highE])


def dct_features(img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Extract DCT-based frequency features from an image.
    
    Args:
        img: Input image array
        cfg: Config dict with 'size' and 'zones' (list of zone boundaries)
        
    Returns:
        Array of DCT features (9 values: 3 stats × 3 regions)
    """
    size = cfg.get('size', 256)
    zones = cfg.get('zones', [0.25, 0.5])  # Default: [low/mid, mid/high]
    
    # Convert to grayscale and resize
    if len(img.shape) == 3:
        gray = color.rgb2gray(img)
    else:
        gray = img
    gray = resize(gray, (size, size), anti_aliasing=True)
    
    # Apply 2D DCT
    dct_full = dct(dct(gray.T, norm='ortho').T, norm='ortho')
    
    # Extract frequency regions based on zones
    h, w = gray.shape
    zone1_bound = int(zones[0] * min(h, w))
    zone2_bound = int(zones[1] * min(h, w))
    
    low = dct_full[:zone1_bound, :zone1_bound]
    mid = dct_full[zone1_bound:zone2_bound, zone1_bound:zone2_bound]
    high = dct_full[zone2_bound:, zone2_bound:]
    
    def stats(x):
        """Calculate statistical features for a frequency region."""
        return [np.mean(np.abs(x)), np.std(x), np.max(np.abs(x))]
    
    return np.array(stats(low) + stats(mid) + stats(high))


def freq_filter_features(img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Extract variance features using frequency filtering.
    
    Args:
        img: Input image array
        cfg: Config dict (currently unused, but kept for consistency)
        
    Returns:
        Array of filter features (2 values: low-pass and high-pass variance)
    """
    size = cfg.get('size', 256)
    
    # Convert to grayscale and resize
    if len(img.shape) == 3:
        gray = color.rgb2gray(img)
    else:
        gray = img
    gray = resize(gray, (size, size), anti_aliasing=True)
    
    # Apply Gaussian low-pass filter
    low_pass = gaussian_filter(gray, sigma=3)
    
    # Extract high-frequency components
    high_pass = gray - low_pass
    
    return np.array([np.var(low_pass), np.var(high_pass)])


def resize_features(img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Simple resize feature (returns empty array, resize is handled in preprocessing).
    This is a placeholder for consistency with config structure.
    
    Args:
        img: Input image array
        cfg: Config dict with 'size'
        
    Returns:
        Empty array (resize is handled separately in data pipeline)
    """
    return np.array([])


def extract_features(image: Union[np.ndarray, torch.Tensor], cfg: Dict[str, Any]) -> torch.Tensor:
    """
    Extract features from an image based on config.
    
    This function reads which feature modules are enabled from cfg.features
    and calls the appropriate extractors.
    
    Args:
        image: Input image (numpy array or torch tensor)
        cfg: Configuration dictionary with 'features' key
        
    Returns:
        Concatenated feature vector as torch tensor [num_features]
    """
    # Convert torch tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        if image.is_cuda:
            image = image.cpu()
        image = image.numpy()
        # Handle CHW format
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
    
    # Ensure image is in [0, 1] range
    if image.max() > 1.0:
        image = image / 255.0
    
    features = []
    
    # Extract enabled features
    feature_configs = cfg.get('features', {})
    
    if feature_configs.get('resize', {}).get('enabled', False):
        feat = resize_features(image, feature_configs.get('resize', {}))
        if len(feat) > 0:
            features.append(feat)
    
    if feature_configs.get('fft', {}).get('enabled', False):
        feat = fft_features(image, feature_configs.get('fft', {}))
        features.append(feat)
    
    if feature_configs.get('dct', {}).get('enabled', False):
        feat = dct_features(image, feature_configs.get('dct', {}))
        features.append(feat)
    
    if feature_configs.get('freq_filter', {}).get('enabled', False):
        feat = freq_filter_features(image, feature_configs.get('freq_filter', {}))
        features.append(feat)
    
    # Concatenate all features
    if len(features) == 0:
        return torch.tensor([], dtype=torch.float32)
    
    combined = np.concatenate(features)
    return torch.tensor(combined, dtype=torch.float32)


def compute_num_features(cfg: Dict[str, Any]) -> int:
    """
    Compute the total number of features that will be extracted based on config.
    
    This is useful for initializing model layers that depend on feature dimension.
    
    Args:
        cfg: Configuration dictionary with 'features' key
        
    Returns:
        Total number of features
    """
    num_features = 0
    feature_configs = cfg.get('features', {})
    
    if feature_configs.get('fft', {}).get('enabled', False):
        num_features += 3  # low, mid, high frequency energy
    
    if feature_configs.get('dct', {}).get('enabled', False):
        num_features += 9  # 3 stats × 3 regions
    
    if feature_configs.get('freq_filter', {}).get('enabled', False):
        num_features += 2  # low-pass and high-pass variance
    
    return num_features


def save_features_cache(features: torch.Tensor, cache_path: Union[str, Path]) -> None:
    """Save features to cache file."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, cache_path)


def load_features_cache(cache_path: Union[str, Path]) -> torch.Tensor:
    """Load features from cache file."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    return torch.load(cache_path)
