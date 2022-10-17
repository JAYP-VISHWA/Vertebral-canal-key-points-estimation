import os
import shutil
def data_dir_prep(config_dict):
  """
  This function prepare data for usage
  
  parameters:
      -config_dict: configuration dictionary as created from YAML file

  return:
      -config_dict: updated for original and tagged image directories
  """
  try:
    shutil.unpack_archive(config_dict["DATA_ZIP_PATH"], config_dict["WORKING_DIR"])
    config_dict["TAGGED_IMAGE_DIR"] = config_dict["WORKING_DIR"]+"/assignment_dataset/tagged_images/"
    config_dict["ORIGINAL_IMAGE_DIR"] = config_dict["WORKING_DIR"]+"/assignment_dataset/original_images"
    return config_dict
  except:
    print("Not able to create directory structure. Problem in data_dir_prep.")