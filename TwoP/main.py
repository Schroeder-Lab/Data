import yaml
import argparse
import pandas as pd

from TwoP.runners import preprocess


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_definitions(path):
    return pd.read_csv(
        path,
        dtype={
            "Name": str,
            "Date": str,
            "Zstack_folder": str,
            "Ignore_planes": str,
            "Process": bool,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess TwoP data')
    parser.add_argument('--config', type=str,
                        required=False, default='zstack.yaml',
                        help='Path to zstack.yaml')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    definitions = load_definitions(config['definitions'])
    preprocess(config, definitions)
