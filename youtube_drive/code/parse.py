import argparse
parser = argparse.ArgumentParser(description=globals()['__doc__'])
parser.add_argument('--config', type=str, required=True, help='Path to the config file') # Path to the config file
args = parser.parse_args()
print(args)