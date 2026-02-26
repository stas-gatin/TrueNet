# TrueNet: AI vs Real Image Detection

TrueNet is an advanced hybrid deep learning system designed to detect and classify images as either **Real** or **AI-Generated**. It utilizes a frequency-aware neural network architecture (`FrequencyAwareAFFNet`), combining Convolutional Neural Network (CNN) backbones with manual, hand-crafted feature extraction to make highly accurate predictions.

## 🌟 Key Features

- **Hybrid AI Detection Architecture:** Uses deep CNN feature maps combined with custom hand-crafted 14-dimensional manual features (like frequency-domain features).
- **Configurable Pipelines:** YAML-based configuration system allows for rapid experimentation with different model architectures, training regimes, and feature extraction mechanisms.
- **Easy Inference:** Simple command-line interface for analyzing individual images.
- **Automated Training Loop:** Supports iterating over multiple experiment configurations sequentially.

## 📁 Repository Structure

```
TrueNet-1/
├── checkpoints/        # Saved model weights (*_best.pth)
├── configs/            # YAML configurations (e.g., exp23.yaml, features_pipeline.yaml)
├── data/               # Datasets and loaders
├── features/           # Hand-crafted manual feature extraction logic
├── models/             # PyTorch model definitions (FrequencyAwareAFFNet, CNN backbones)
├── pipelines/          # Training, testing, and evaluation loops (Trainer)
├── utils/              # Helper functions and utilities
├── main.py             # Core execution entry point for training and inference
├── predict.py          # Prediction and classification logic
└── README.md           # Project documentation
```

## 🚀 Usage

### 1. Running Inference (Image Prediction)

To analyze a single image and determine if it is Real or AI-Generated, use the `--image` flag with `main.py`:

```bash
python main.py \
    --image /path/to/your/image.jpg \
    --config configs/exp23.yaml \
    --features configs/features_pipeline.yaml
```

**Expected Output:**
```
Executing prediction on image: /path/to/your/image.jpg
Model FrequencyAwareAFFNet loaded successfully.
Analysis Results: Predicted Class: AI-Generated, AI Probability: 0.9824
```

### 2. Training Models

To train the models based on predefined configuration files, simply execute `main.py` without any arguments. The script will automatically iterate through the experiments defined in `main.py`'s `experiments` list.

```bash
python main.py
```

*Note: You may need to modify the `experiments` list inside `main.py` to point to valid configuration files like `configs/exp23.yaml` instead of missing configs.*

### 3. Testing Predictions

You can also use the `test_predict.py` script to quickly test an image prediction by modifying a script instead of passing command-line arguments.

1. Open `test_predict.py`.
2. Modify the `IMAGE_PATH` variable (line 8) to point to your image: `IMAGE_PATH = "path/to/your/test/image.jpg"`.
3. Run the script:
```bash
python test_predict.py
```

## ⚙️ Configuration Files

Configurations are housed in the `configs/` directory.

- **`exp*.yaml`**: Primary configuration containing dataset paths, model preferences (CNN backbone, feature dimension), and training hyperparameters (batch size, learning rate, freeze layers).
- **`features_pipeline.yaml`**: Dedicated configuration governing the extraction process for the manual/hand-crafted image features. Contains normalization parameters (`imagenet_mean`, `imagenet_std`) and input sizing.

## 📝 Rules & Guidelines
- All work is isolated within this current workspace.
- The default model checkpoints will be searched for inside the `checkpoints/` directory as named by the `best_model_name` string in the active config.

---

*This system was developed as part of advanced image analysis efforts emphasizing synthetic document and image detection.*