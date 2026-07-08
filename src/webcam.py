import argparse

from .detect import run_inference_on_webcam
from .utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run webcam inference")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_inference_on_webcam(config)


if __name__ == "__main__":
    main()
