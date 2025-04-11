import argparse
import yaml

def load_config(path="configs/environments.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def str_to_type(type_str):
    return {'int': int, 'float': float, 'str': str, 'bool': bool}[type_str]

def build_parser(config):
    parser = argparse.ArgumentParser(description="Gym4ReaL: Launch Environment")
    subparsers = parser.add_subparsers(dest="env", required=True, help="Choose environment")

    for env_name, env_info in config.items():
        env_parser = subparsers.add_parser(env_name, help=env_info.get("description", ""))
        for arg in env_info.get("arguments", []):
            kwargs = {
                "help": arg.get("help", ""),
                "default": arg.get("default"),
                "type": str_to_type(arg.get("type", "str")),
            }
            env_parser.add_argument(arg["name"], **kwargs)
    return parser

if __name__ == "__main__":
    config = load_config()
    parser = build_parser(config)
    args = parser.parse_args()
    print(args)
