import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from data.dataset import ChunkedFeatureDataset
from models.cnn_detector import HybridRealFakeCNN

def find_best_f1_threshold(model_path: str, cache_dir: str, batch_size: int = 64, device=None) -> tuple:
    """
    Loads saved model, computes probabilities, and calculates the optimal F1 threshold.
    
    Args:
        model_path (str): File path representing model state dictionary snapshot.
        cache_dir (str): Cached precomputed components destination directory string.
        batch_size (int): Size per forward evaluation run chunks iterations loop limit integer.
        device (Device): Selected processor target mapping node interface.
    
    Returns:
        tuple: (best_threshold, best_f1, probability list bounds arrays format representation, target predictions ground truth matches representations).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")

    # Load compiled inference ready model definitions
    model = HybridRealFakeCNN(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load pipeline integration source map points references bounds target chunks cache logic dataset 
    dataset = ChunkedFeatureDataset(cache_dir=cache_dir, transform=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, feats, labels in loader:
            imgs = imgs.to(device)
            feats = feats.to(device) if feats is not None else None
            outputs = model(imgs, feats)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    # Check available target evaluation bounds range 0.1 mapping constraint definitions layout thresholds 
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.91, 0.01):
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"Best recorded F1 score value map definition representation result returned is bounds target values match parameters: {best_f1:.4f} found at specific mapped configuration definition layout variable threshold {best_thresh:.2f}")
    return best_thresh, best_f1, y_prob, y_true

# Logic validation block testing
if __name__ == "__main__":
    model_path = "checkpoints/hybrid_best_60k_chank16_batch64_best.pth"
    cache_dir = "features/cache/test"  # Alternatively using validation reference points map layout location folder config directory references paths validation
    best_thresh, best_f1, probs, labels = find_best_f1_threshold(model_path, cache_dir)
