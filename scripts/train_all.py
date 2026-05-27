import argparse
from src.cyberbully_detector.train import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-frac", type=float, default=1.0)
    args = parser.parse_args()
    best = run_training(sample_frac=args.sample_frac)
    print("Best model:", best["name"], "type:", best["type"], "f1:", best["metrics"].get("f1_macro"))
