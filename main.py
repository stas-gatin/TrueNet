import argparse
from pipelines.train_pipeline import Trainer
from predict import predict_image

# List of experiment configuration files to run sequentially
experiments = [
    "configs/exp24.yaml",
]

def train_models():
    """
    Executes training loops over multiple experiment configurations.
    """
    for cfg in experiments:
        trainer = Trainer(cfg)
        trainer.train() # Train the model
        #trainer.test() # Test after training, only for already existing models
        trainer.evaluate() # Evaluate after testing, generates results, plots, and metrics

def main():
    """Main entry point for TrueNet execution."""
    parser = argparse.ArgumentParser(description="TrueNet Core Execution Script")
    parser.add_argument("--image", type=str, help="Absolute path to the image for prediction")
    parser.add_argument("--config", type=str, default="configs/exp23.yaml", help="Path to the model configuration file")
    parser.add_argument("--features", type=str, default="configs/features_pipeline.yaml", help="Path to the feature extraction configuration file")
    args = parser.parse_args()

    if args.image:
        print(f"Executing prediction on image: {args.image}")
        prob_ai, predicted_class, predicted_name = predict_image(args.image, args.config, args.features)
        print(f"Analysis Results: Predicted Class: {predicted_name}, AI Probability: {prob_ai:.4f}")
    else:
        print("Starting training process for listed configurations...")
        train_models()

if __name__ == "__main__":
    main()