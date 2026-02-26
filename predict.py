import os
import yaml
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np
from typing import Dict, Any, Tuple
import torch.nn.functional as F
from features.features import extract_features
from copy import deepcopy 

def _load_yaml(path: str) -> Dict:
    """Load yaml configurations"""
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}

def pil_to_rgb(img: Image.Image) -> Image.Image:
    """Converts a PIL Image object to RGB"""
    if img.mode in ("P", "LA", "PA"):
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        return background
    return img

def _build_inference_transforms(data_cfg: Dict, train_cfg: Dict) -> transforms.Compose:
    """Creates transforms for inference."""
    mean = data_cfg.get('imagenet_mean', [0.485, 0.456, 0.406])
    std = data_cfg.get('imagenet_std', [0.229, 0.224, 0.225])
    input_size = int(data_cfg.get('input_size', 224))

    t = [transforms.Lambda(pil_to_rgb),
         transforms.Resize((input_size, input_size)),
         transforms.ToTensor(), 
         transforms.Normalize(mean=mean, std=std)]
    
    return transforms.Compose(t)

from models.cnn_detector import FrequencyAwareAFFNet # Import the best model architecture

def predict_image(image_path: str, cfg_path: str, features_cfg_path: str = 'features_pipeline.yaml') -> Tuple[float, int, str]:
    """
    Loads saved layouts and performs a single evaluation.
    
    Args:
        image_path (str): The absolute path to the image to test.
        cfg_path (str): The absolute path to the experiment configuration.
        features_cfg_path (str): The absolute path to the feature configurations. 
    
    Returns:
        tuple: (prob_ai, predicted_class, predicted_name)
    """
    
    # 1. Load configuration
    try:
        cfg = _load_yaml(cfg_path)
        features_cfg = _load_yaml(features_cfg_path)
        
        # Merge configuration
        temp_cfg = deepcopy(cfg)
        feature_config_data = features_cfg.get('feature_config', {})
        if feature_config_data:
            temp_cfg.update(feature_config_data)
        
        cfg = temp_cfg 
        
    except Exception as e:
        print(f"Error loading yaml configuration: {e}")
        return 0.0, 0, "Error"
        
    data_cfg = cfg.get('data', {}) 
    data_cfg['imagenet_mean'] = features_cfg.get('imagenet_mean', data_cfg.get('imagenet_mean'))
    data_cfg['imagenet_std'] = features_cfg.get('imagenet_std', data_cfg.get('imagenet_std'))
    data_cfg['input_size'] = features_cfg.get('input_size', data_cfg.get('input_size')) 
    
    model_cfg = cfg.get('model', {})
    train_cfg = cfg.get('training', {})
    
    model_name = model_cfg.get('model_name')
    default_best = f"{model_name}_best.pth" if model_name else 'hybrid_best.pth'
    best_model_name = model_cfg.get('best_model_name', default_best)
    checkpoint_dir = model_cfg.get('checkpoint_dir', 'checkpoints')
    best_model_path = os.path.join(checkpoint_dir, best_model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    backbone_name = model_cfg.get('cnn_backbone', 'efficientnet_b0')
    hand_dim = model_cfg.get('hand_feat_dim', 14) 
    pretrained = bool(model_cfg.get('pretrained', True))
    freeze_until = float(train_cfg.get('freeze_until', 0.5))
    dropout_rate = float(train_cfg.get('dropout_rate', 0.7))
    
    try:
        model = FrequencyAwareAFFNet(
            backbone_name=backbone_name, 
            pretrained=pretrained, 
            hand_feat_dim=hand_dim, 
            freeze_until=freeze_until, 
            dropout_rate=dropout_rate
        ).to(device)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return 0.0, 0, "Error"

    if not os.path.exists(best_model_path):
        print(f"Error: Model checkpoint not found at: {best_model_path}")
        return 0.0, 0, "Error"
        
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    print(f"Model {model_name} loaded successfully.")

    # 2. Extract image properties
    try:
        image = Image.open(image_path).convert('RGB')
        img_np = np.array(image) 
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return 0.0, 0, "Error"

    # --- Extract manual features ---
    hand_feats_tensor = extract_features(img_np, cfg).unsqueeze(0).to(device) # [1, 14]
    
    if hand_feats_tensor.shape[1] != hand_dim:
        print(f"Error: Number of manual features ({hand_feats_tensor.shape[1]}) does not match model hand_dim ({hand_dim}). Check {features_cfg_path}!")
        return 0.0, 0, "Error"
    
    # --- Prepare CNN transforms ---
    transform = _build_inference_transforms(data_cfg, train_cfg)
    cnn_input_tensor = transform(image).unsqueeze(0).to(device) # [1, 3, H, W]


    # 3. Model forward pass
    with torch.no_grad():
        outputs = model(cnn_input_tensor, hand_feats_tensor)

    # 4. Evaluate probabilities
    probs = F.softmax(outputs, dim=1)[0]
    prob_ai = probs[1].item() 
    
    predicted_class = 1 if prob_ai >= 0.5 else 0
    
    class_names = model_cfg.get('class_names', ['Real', 'AI-Generated'])
    predicted_name = class_names[predicted_class]

    return prob_ai, predicted_class, predicted_name
