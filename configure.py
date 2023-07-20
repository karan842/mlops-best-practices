import os
import sys
import yaml

def read_configure():
    yaml_path = os.path.join(os.path.dirname(__file__),'configure.yaml')
    with open(yaml_path, 'r') as yaml_file:
        configure = yaml.safe_load(yaml_file)
    return configure

configure = read_configure()

if __name__ == '__main__':
    print("You can import configure straight as a module")