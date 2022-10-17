import yaml

def get_configuration_dict(config_yaml_path):
  """
  This function read configuration YAML file and provide dictionary with configuration

  parameters:
      -config_yaml_path: path of the yaml file having configuration

  return:
      -config_dict: diction having key- value pairs for configuration
  """
  with open(config_yaml_path, "r") as stream:
      try:
          config_dict = yaml.safe_load(stream)
          return config_dict
      except yaml.YAMLError as exc:
          print(exc)

