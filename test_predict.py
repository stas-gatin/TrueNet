from predict import predict_image

def main():
    """
    A simple script to test an image prediction by modifying the IMAGE_PATH variable.
    """
    # Replace this with the actual path to the image you want to test
    IMAGE_PATH = "path/to/your/test/image.jpg"
    
    # The configuration paths used for model and features
    MODEL_CONFIG = "configs/exp23.yaml"
    FEATURES_CONFIG = "configs/features_pipeline.yaml"
    
    print(f"Testing prediction on image: {IMAGE_PATH}")
    print("Loading model, extracting features, and making a prediction...")
    
    try:
        prob_ai, predicted_class, predicted_name = predict_image(
            IMAGE_PATH, 
            MODEL_CONFIG, 
            FEATURES_CONFIG
        )
        
        print("\n" + "="*40)
        print("PREDICTION RESULTS:")
        print("="*40)
        print(f"Predicted Class: {predicted_name} (Class id: {predicted_class})")
        print(f"AI Probability:  {prob_ai:.4f} ({prob_ai * 100:.2f}%)")
        print("="*40)
        
    except FileNotFoundError:
        print(f"\n[ERROR] Could not find the image file at: {IMAGE_PATH}")
        print("Please update the IMAGE_PATH variable with a valid image path.")
    except Exception as e:
        print(f"\n[ERROR] An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()
