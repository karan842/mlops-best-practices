import yaml

yaml_path = './configure.yaml'

with open(yaml_path, 'r') as yaml_file:
    data = yaml.safe_load(yaml_file)

src = data['data_source']['type']

print(src)