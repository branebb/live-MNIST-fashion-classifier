import argparse
from model_pipeline.train import train_model
from model_pipeline.model_utils import load_model
from model_pipeline.dataset import test_loader

def main() -> None:
    """Main function to run training or inference."""
    parser = argparse.ArgumentParser(description="FashionMNIST Model Runner")
    parser.add_argument(
        "mode", 
        choices=["train", "predict"], 
        help="Mode: 'train' or 'predict'"
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "predict":
        model = load_model()
        print(f"Model has {model.num_parameters} parameters.")
        sample_image, sample_label = test_loader.dataset[420]
        print(sample_image.shape)
        predicted_class, probabilities = model.predict(sample_image)
        
        print(f"Predicted class: {predicted_class}, "
              f"Actual class: {sample_label}")
        print(f"Class probabilities: {probabilities}")

if __name__ == "__main__":
    main()