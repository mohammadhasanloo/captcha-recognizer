"""Command line entry point: ``python -m captcha_recognizer.cli ...``"""

from __future__ import annotations

import argparse
from pathlib import Path

import keras

from captcha_recognizer.data import load_dataset, train_test_split
from captcha_recognizer.demo import prediction_grid, write_metrics, write_predictions_csv
from captcha_recognizer.evaluate import predict_texts, score
from captcha_recognizer.train import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_SEED, train

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLES = ROOT / "samples"
DEFAULT_MODEL = ROOT / "captcha_recognizer.keras"
DEFAULT_DOCS = ROOT / "docs"
DEFAULT_RESULTS = ROOT / "results"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    subparsers = parser.add_subparsers(dest="command", required=True)

    trainer = subparsers.add_parser("train", help="train, evaluate and write figures")
    trainer.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    trainer.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    subparsers.add_parser("evaluate", help="score a saved model on the held-out split")
    predictor = subparsers.add_parser("predict", help="read specific captcha images")
    predictor.add_argument("images", nargs="+", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "train":
        train(
            samples_dir=args.samples,
            model_path=args.model,
            docs_dir=args.docs,
            results_dir=args.results,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        return 0

    dataset, vocabulary = load_dataset(args.samples, seed=args.seed)
    model = keras.models.load_model(args.model)

    if args.command == "evaluate":
        _, test_set = train_test_split(dataset, train_fraction=0.8)
        scores = score(model, test_set)
        print(scores.format())
        prediction_grid(model, test_set, vocabulary, args.docs / "demo.png")
        write_metrics(scores, args.results / "metrics.json")
        write_predictions_csv(model, test_set, vocabulary, args.results / "predictions.csv")
        return 0

    if args.command == "predict":
        import numpy as np

        from captcha_recognizer.data import IMAGE_HEIGHT, IMAGE_WIDTH, denoise

        import cv2

        images = np.zeros((len(args.images), IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)
        for i, path in enumerate(args.images):
            grey = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if grey is None:
                raise SystemExit(f"could not read {path}")
            images[i, :, :, 0] = denoise(grey) / 255.0
        for path, text in zip(args.images, predict_texts(model, images, vocabulary)):
            print(f"{path}: {text}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
