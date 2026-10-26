import logging
from house_price_predictor.train import train


def main() -> None:
    """Configure logging and run the training workflow."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    train()


if __name__ == "__main__":
    main()
