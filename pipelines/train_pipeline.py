"""
Training pipeline wrapper used by `main.py`.
Provides a `Trainer` class that loads an experiment config (YAML),
...
"""
import os
import yaml
import json
import time
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
from copy import deepcopy
import torch.optim as optim
from torchvision import transforms
from typing import Dict, Tuple, Any
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from data.dataset import ChunkedFeatureDataset
from utils.telegram_notify import send_message
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, precision_recall_curve, auc, classification_report, f1_score
from models.cnn_detector import HybridRealFakeCNN, ImprovedHybridCNN, HybridRealFakeCNN_Normalized, HybridRealFakeCNN_HeavyDrop, AugmentedFeatureFusionNet, FrequencyAwareAFFNet

# IMPORTS FOR GOOGLE SHEETS
try:
    import gspread
    from utils.gsheets_logger import connect_to_gsheet, log_experiment
    GSHEETS_AVAILABLE = True
    print("Google Sheets logging is available.")
except ImportError:
    GSHEETS_AVAILABLE = False
    print("Warning: gspread/gsheets_logger not found. Google Sheets logging disabled.")

def _load_yaml(path: str) -> Dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}

def _deep_update(base: Dict, override: Dict) -> Dict:
    """Recursively update `base` with `override` and return result."""
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = deepcopy(v)
    return base

def pil_to_rgb(img):
    """
    Convert PIL Image to RGB format, handling transparency and palette modes.
    Ensures all images have a consistent RGB format by:
    - Converting palette-based images (P mode) to RGBA first
    - Converting grayscale + alpha (LA) to RGBA
    - Converting palette + alpha (PA) to RGBA
    - Replacing transparency with white background
    This preprocessing is critical for consistent model input during training.
    Args:
        img (PIL.Image): Input image in any PIL-supported format.
    Returns:
        PIL.Image: RGB image with white background replacing transparency.
    Notes:
    - Used in both train and inference pipelines for consistency
    - White background (255, 255, 255) preserves image content visibility
    """
    # Convert palette/transparency modes to RGBA for consistency
    if img.mode in ("P", "LA", "PA"):
        img = img.convert("RGBA")
    # Replace alpha channel with white background
    if img.mode == "RGBA":
        # Create white RGB background
        background = Image.new("RGB", img.size, (255, 255, 255))
        # Composite image using alpha channel as mask
        background.paste(img, mask=img.split()[-1])
        return background
    return img

def tensor_augment(img: torch.Tensor):
    if img.max() > 1.0:
        img = img.float() / 255.0
    if random.random() > 0.5:
        img = TF.hflip(img)
    return img

class TensorAugmentations:
    def __init__(self, input_size=224, p_gray=0.05):
        self.input_size = input_size
        self.p_gray = p_gray

    def __call__(self, img: torch.Tensor):
        # img: [C,H,W]
        return tensor_augment(img)

class Trainer:
    """Experiment trainer used by `main.py`.
    Args:
        cfg_path: Path to an experiment YAML config which overrides
            `configs/config.yaml` defaults.
    """
    def __init__(self, cfg_path: str):
        # Total training time tracker
        self.total_time = 0.0
        # Config path
        self.cfg_path = cfg_path
        # Load base defaults and override with experiment config
        base_cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
        # allow relative path from repo root as well
        repo_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        alt_base = os.path.join(repo_base, 'configs', 'config.yaml')
        if os.path.exists(base_cfg_path):
            base_cfg = _load_yaml(base_cfg_path)
        elif os.path.exists(alt_base):
            base_cfg = _load_yaml(alt_base)
        else:
            base_cfg = {}

        exp_cfg = _load_yaml(cfg_path) if cfg_path and os.path.exists(cfg_path) else {}
        cfg = _deep_update(base_cfg, exp_cfg)

        # Expose merged config
        self.cfg = cfg

        # Shorthand sections with safe defaults
        self.data_cfg = cfg.get('data', {})
        self.model_cfg = cfg.get('model', {})
        self.train_cfg = cfg.get('training', {})
        self.eval_cfg = cfg.get('evaluation', {})
        self.device_cfg = cfg.get('device', {})
        
        # === GOOGLE SHEETS SETUP ===
        self.gsheets_sheet = None
        if GSHEETS_AVAILABLE:
            gsheets_cfg = cfg.get('gsheets', {})
            creds_path = gsheets_cfg.get('credentials_path')
            sheet_name = gsheets_cfg.get('sheet_name')
            worksheet_index = int(gsheets_cfg.get('worksheet_index', 0))
            
            if creds_path and sheet_name:
                self.gsheets_sheet = connect_to_gsheet(creds_path, sheet_name, worksheet_index)
        # ===========================

        # Derived/expected paths
        cache_dir = self.data_cfg.get('cache_dir', 'data/cache')
        self.train_cache = self.data_cfg.get('train_cache_file', os.path.join(cache_dir, 'train_features.pt'))
        self.test_cache = self.data_cfg.get('test_cache_file', os.path.join(cache_dir, 'test_features.pt'))
        self.results_dir = self.data_cfg.get('results_dir', 'results')
        self.checkpoint_dir = self.model_cfg.get('checkpoint_dir', 'checkpoints')

        # If a model_name is provided, use it to build a default best-model filename
        model_name = self.model_cfg.get('model_name')
        if model_name:
            default_best = f"{model_name}_best.pth"
        else:
            default_best = 'hybrid_best.pth'
        self.best_model_name = self.model_cfg.get('best_model_name', default_best)

        # Place results under model-specific subfolder when model_name is set
        if model_name:
            self.results_dir = os.path.join(self.results_dir, model_name)
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Path for training log
        self.train_log_path = os.path.join(self.results_dir, 'training_log.csv')

        # If the file does not exist, create the headers
        if not os.path.exists(self.train_log_path):
            with open(self.train_log_path, 'w') as f:
                f.write('epoch,train_loss,val_loss,val_auc,val_accuracy,val_f1,epoch_time\n')


        # Device selection (simple heuristic)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')

        # Instantiate model (pass hand_feat_dim if present)
        backbone_name = self.model_cfg.get('cnn_backbone', '')
        hand_dim = self.model_cfg.get('hand_feat_dim', 14)
        pretrained = bool(self.model_cfg.get('pretrained', True))
        # Dictionary mapping model name to the actual class constructor
        model_mapping = {
            'HybridRealFakeCNN': HybridRealFakeCNN,
            'ImprovedHybridCNN': ImprovedHybridCNN,
            'HybridRealFakeCNN_Normalized': HybridRealFakeCNN_Normalized,
            'HybridRealFakeCNN_HeavyDrop': HybridRealFakeCNN_HeavyDrop,
            'AugmentedFeatureFusionNet': AugmentedFeatureFusionNet,
            'FrequencyAwareAFFNet': FrequencyAwareAFFNet
        }

        # Obtain and instantiate model type from configurations
        model_type = self.model_cfg.get('model_type', 'HybridRealFakeCNN')
        freeze_until = float(self.train_cfg.get('freeze_until', 0.8))
        dropout_rate = float(self.train_cfg.get('dropout_rate', 0.6))

        # Initialize the chosen model along with specific hyper parameters
        model_cls = model_mapping.get(model_type)
        if model_cls is None:
            raise ValueError(f"Unknown model type '{model_type}'")
            
        self.model = model_cls(backbone_name=backbone_name, pretrained=pretrained, hand_feat_dim=hand_dim, freeze_until=freeze_until, dropout_rate=dropout_rate).to(self.device)


        # Training hyperparameters
        self.batch_size = int(self.train_cfg.get('train_batch_size', 128))
        self.test_batch_size = int(self.train_cfg.get('test_batch_size', self.batch_size))
        self.num_epochs = int(self.train_cfg.get('num_epochs', 10))
        self.lr = float(self.train_cfg.get('learning_rate', 1e-4))
        self.num_workers = int(self.train_cfg.get('num_workers', 4))
        self.patience = int(self.train_cfg.get('early_stopping_patience', 3))
        
        # Checkpoint path
        self.best_model_path = os.path.join(self.checkpoint_dir, self.best_model_name)

    def _build_transforms(self, train: bool = True):
    # ... (method remains unchanged) ...
        # Keep preprocessing consistent with training/test scripts
        mean = self.data_cfg.get('imagenet_mean', [0.485, 0.456, 0.406])
        std = self.data_cfg.get('imagenet_std', [0.229, 0.224, 0.225])
        input_size = int(self.data_cfg.get('input_size', 224))

        t = [transforms.Lambda(pil_to_rgb),
             transforms.Resize((input_size, input_size))]

        if train:
            if float(self.train_cfg.get('augmentation_flip_prob', 0.0)) > 0:
                from torchvision import transforms as T
                t.append(T.RandomHorizontalFlip(p=float(self.train_cfg.get('augmentation_flip_prob', 0.5))))
            
        t.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        
        print(f"{'Train' if train else 'Test'} transforms: {t}")
        return transforms.Compose(t)

    def train(self):
    # ... (method remains unchanged) ...
        """Train the model and save best checkpoint by validation AUC."""
        print(f"Starting training: epochs={self.num_epochs}, batch={self.batch_size}, device={self.device}")

        # NOTE: If cache contains precomputed 'images' tensors, transforms may be skipped.
        transform = self._build_transforms(train=True)
        train_ds = ChunkedFeatureDataset(
            cache_dir='features/cache/train',
            #transform=TensorAugmentations(input_size=224)
        )

        #train_ds = ChunkedFeatureDataset(cache_dir=self.data_cfg.get('train_cache_dir', 'features/cache/train'),
        #                                 transform=transform)
        val_ds = ChunkedFeatureDataset(cache_dir=self.data_cfg.get('test_cache_dir', 'features/cache/test'))

        # After dataset loading, verify features shape and alert if unexpected
        try:
            sample_feat = train_ds[0][1] # get features of first element
            print(f"Feature vector shape (train): {tuple(sample_feat.shape)}")
        except Exception:
            pass

        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  pin_memory=self.device.type == 'cuda', num_workers=self.num_workers)
        val_loader = DataLoader(val_ds, batch_size=self.test_batch_size, shuffle=False,
                                pin_memory=self.device.type == 'cuda', num_workers=max(0, self.num_workers))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

        best_auc = 0.0
        wait = 0

        total_start = time.time()

        for epoch in range(self.num_epochs):
            start_time = time.time()
            self.model.train()
            running_loss = 0.0
            
            for cnn_input, hand_feats, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [TRAIN]", leave=False):
                cnn_input = cnn_input.to(self.device)
                labels = labels.to(self.device)
                hand_feats = hand_feats.to(self.device) if hand_feats is not None else None

                optimizer.zero_grad()
                outputs = self.model(cnn_input, hand_feats)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * cnn_input.size(0)

            epoch_loss = running_loss / max(1, len(train_ds))

            # Validation AUC and Accuracy
            val_loss, val_auc, val_accuracy, val_f1 = self._validate_metrics(val_loader)

            epoch_time = time.time() - start_time  # epoch time

            print(f"Epoch {epoch+1} — train_loss: {epoch_loss:.4f} | val_loss: {val_loss:.4f} | val_auc: {val_auc:.4f} | val_acc: {val_accuracy:.4f} | val_f1: {val_f1:.4f}")

            # Log to CSV
            with open(self.train_log_path, 'a') as f:
                f.write(f"{epoch+1},{epoch_loss:.6f},{val_loss:.6f},{val_auc:.6f},{val_accuracy:.6f},{val_f1:.6f},{epoch_time:.2f}\n")

            # Checkpointing based on AUC
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(self.model.state_dict(), self.best_model_path)
                wait = 0
                print(f"New best AUC: {best_auc:.4f} — saved: {self.best_model_path}")
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"Early stopping (no improvement for {self.patience} epochs)")
                    break
        
        self.total_time = time.time() - total_start
        print("Training finished.")
        print(f"Total training time: {self.total_time:.1f}s ({self.total_time/60:.2f} min)")

    def _validate_metrics(self, loader: DataLoader) -> Tuple[float, float, float, float]:
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        all_probs = []
        all_labels = []
        total_loss = 0.0
        n_samples = 0
        conf_threshold = float(self.eval_cfg.get('confidence_threshold', 0.5))

        with torch.no_grad():
            for cnn_input, hand_feats, labels in loader:
                cnn_input = cnn_input.to(self.device)
                hand_feats = hand_feats.to(self.device) if hand_feats is not None else None
                labels = labels.to(self.device)
                
                outputs = self.model(cnn_input, hand_feats)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * cnn_input.size(0)
                n_samples += cnn_input.size(0)

                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        val_loss = total_loss / max(1, n_samples)
        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)
        y_pred = (y_prob >= conf_threshold).astype(int)

        try:
            auc_score = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc_score = 0.0

        try:
            accuracy = float(accuracy_score(y_true, y_pred))
        except Exception:
            accuracy = 0.0

        try:
            f1 = float(f1_score(y_true, y_pred))
        except Exception:
            f1 = 0.0

        return val_loss, auc_score, accuracy, f1


    def test(self, save_results: bool = True) -> Dict:
        """Run inference on test set using the best checkpoint and return metrics.
        Returns a dict with keys: `auc`, `accuracy`, `f1_score`, `confusion_matrix`, `n_samples`.
        """
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
            print(f"Loaded model checkpoint: {self.best_model_path}")
        else:
            print(f"Warning: best model not found at {self.best_model_path}. Using current weights.")

        transform = self._build_transforms(train=False)
        test_ds = ChunkedFeatureDataset(cache_dir=self.data_cfg.get('test_cache_dir', 'features/cache/test'))
        test_loader = DataLoader(test_ds, batch_size=self.test_batch_size, shuffle=False,
                                 pin_memory=self.device.type == 'cuda', num_workers=max(0, self.num_workers))

        self.model.eval()
        all_probs = []
        all_preds = []
        all_labels = []
        conf_threshold = float(self.eval_cfg.get('confidence_threshold', 0.5))

        with torch.no_grad():
            for cnn_input, hand_feats, labels in tqdm(test_loader, desc="Testing", leave=False):
                cnn_input = cnn_input.to(self.device)
                hand_feats = hand_feats.to(self.device) if hand_feats is not None else None
                outputs = self.model(cnn_input, hand_feats)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = (probs >= conf_threshold).astype(int)
                all_probs.extend(probs.tolist())
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.numpy().tolist())

        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)
        y_pred = np.array(all_preds)

        results = {
            'n_samples': int(len(test_ds)),
            'auc': None,
            'accuracy': None,
            'f1_score': None, # Added F1-score
            'confusion_matrix': None
        }

        try:
            results['auc'] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            results['auc'] = None

        results['accuracy'] = float(accuracy_score(y_true, y_pred))
        results['f1_score'] = float(f1_score(y_true, y_pred)) # Calculate and add F1-score
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

        if save_results:
            out_json = os.path.join(self.results_dir, 'test_results.json')
            with open(out_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved test results: {out_json}")
            
        # === GOOGLE SHEETS LOGGING ===
        if self.gsheets_sheet:
            log_experiment(self.gsheets_sheet, self.cfg, results, self.cfg_path, self.total_time)
            print("Logged experiment results to Google Sheets.")

        try:
            send_message(
                f"✅ Training & Evaluation Completed!\n\n"
                f"🖥 Model: {self.model_cfg.get('model_name', 'Unnamed Model')}\n"
                f"📊 Metrics:\n"
                f"   • AUC: {results['auc']:.4f}\n"
                f"   • Accuracy: {results['accuracy']:.4f}\n"
                f"   • F1-score: {results['f1_score']:.4f}\n"
                f"⏱ Elapsed Time: {self.total_time:.1f}s"
            )
        except Exception:
            pass
        # =============================


        # Try plotting if matplotlib + seaborn + pandas available
        try:
        # ... (Code for drawing plots remains unchanged) ...
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            # Prepare class names
            class_names = self.model_cfg.get('class_names', ['Real', 'AI-Generated'])

            # 1) Confusion matrix heatmap
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=class_names, yticklabels=class_names)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Confusion Matrix')
            fig.tight_layout()
            pth = os.path.join(self.results_dir, 'confusion_matrix.png')
            fig.savefig(pth, dpi=self.eval_cfg.get('figure_dpi', 300), bbox_inches='tight')
            plt.close(fig)

            # 2) ROC curve
            try:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(7, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC)')
                plt.legend(loc='lower right')
                plt.grid(alpha=0.3)
                pth = os.path.join(self.results_dir, 'roc_curve.png')
                plt.tight_layout()
                plt.savefig(pth, dpi=self.eval_cfg.get('figure_dpi', 300), bbox_inches='tight')
                plt.close()
            except Exception:
                pass

            # 3) Precision-Recall curve
            try:
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                pr_auc = auc(recall, precision)
                plt.figure(figsize=(7, 6))
                plt.plot(recall, precision, color='green', lw=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
                plt.axhline(y=y_true.mean(), color='gray', linestyle='--', label=f'No Skill ({y_true.mean():.3f})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend(loc='lower left')
                plt.grid(alpha=0.3)
                pth = os.path.join(self.results_dir, 'precision_recall_curve.png')
                plt.tight_layout()
                plt.savefig(pth, dpi=self.eval_cfg.get('figure_dpi', 300), bbox_inches='tight')
                plt.close()
            except Exception:
                pass

            # 4) Probability distribution per class
            try:
                plt.figure(figsize=(8, 5))
                sns.histplot(y_prob[y_true == 0], bins=50, alpha=0.7, label=class_names[0], color='skyblue', kde=True)
                sns.histplot(y_prob[y_true == 1], bins=50, alpha=0.7, label=class_names[1], color='salmon', kde=True)
                plt.xlabel('AI-Generated Probability')
                plt.ylabel('Count')
                plt.title('Prediction Probability Distribution')
                plt.legend()
                pth = os.path.join(self.results_dir, 'probability_distribution.png')
                plt.tight_layout()
                plt.savefig(pth, dpi=self.eval_cfg.get('figure_dpi', 300), bbox_inches='tight')
                plt.close()
            except Exception:
                pass

            # 5) Classification report table -> image
            try:
                cr = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
                report_df = pd.DataFrame(cr).transpose().round(4)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.axis('tight')
                ax.axis('off')
                table = ax.table(cellText=report_df.values, colLabels=report_df.columns, rowLabels=report_df.index,
                                 cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.6)
                plt.title('Classification Report', pad=20)
                pth = os.path.join(self.results_dir, 'classification_report.png')
                plt.tight_layout()
                fig.savefig(pth, dpi=self.eval_cfg.get('figure_dpi', 300), bbox_inches='tight')
                plt.close(fig)
            except Exception:
                pass

            print(f"Saved plots to: {self.results_dir}")
        except Exception:
            # If plotting libraries are not available or any error occurs, skip plotting
            pass
            
        return results

    def evaluate(self):
        """Convenience wrapper: run test() and print a short summary."""
        res = self.test(save_results=True)
        print("\n" + "=" * 60)
        print(f"TEST SUMMARY — samples: {res.get('n_samples')}")
        print(f"AUC: {res.get('auc')}")
        print(f"Accuracy: {res.get('accuracy')}") # Output Accuracy
        print(f"F1-score: {res.get('f1_score')}") # Output F1-score
        print("=" * 60)
        return res