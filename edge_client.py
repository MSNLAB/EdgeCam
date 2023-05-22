import argparse
import munch
import yaml

from edge.edge_worker import EdgeWorker
from loguru import logger





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="configuration description")
    parser.add_argument("--yaml_path", default="./config/config.yaml", help="input the path of *.yaml")
    args = parser.parse_args()
    with open(args.yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    #provide class-like access for dict
    config = munch.munchify(config)
    config = config.client
    edge = EdgeWorker(config)
    logger.add("log/client/client_{time}.log", level="INFO", rotation="500 MB")





