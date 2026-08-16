"""CLI entry point: run the fraud detection pipeline against a dataset folder.

Usage:
    uv run main.py "Brave New World - validation"
"""

import argparse
import asyncio

from src.agents import run_pipeline_async


def main():
    parser = argparse.ArgumentParser(description="Run the multi-agent fraud detection pipeline.")
    parser.add_argument(
        "dataset",
        help="Path to a dataset folder containing transactions.csv, sms.json, mails.json and locations.json",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Number of transactions processed per batch (default: 12)",
    )
    args = parser.parse_args()

    asyncio.run(run_pipeline_async(args.dataset, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
