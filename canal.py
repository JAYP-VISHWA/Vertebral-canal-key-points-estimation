from yaml_reader import get_configuration_dict
from data_prep import data_dir_prep
from reference_feat import prep_reference_df, reference_feature_extraction
from utils import image_enhancement, euclidean_distance
from point_selection import *
import glob

def create_reference_files(yaml_file_path):
  """
  This fundtion works as main funtion for reference creation
  """
  config_dict = get_configuration_dict(yaml_file_path)
  config_dict = data_dir_prep(config_dict)
  df_ref = prep_reference_df(config_dict)
  reference_feature_extraction(config_dict, df_ref)
  return 


def prep_test_df(config_dict):
  """
  This function prepares test dataframe
  
  parameters:
      -config_dict: configuration dictionary

  return:
      -df_reference: dataframe with original and tagged file paths and corresponding reference start and end points

  """
  list_original_image_files = glob.glob(config_dict["ORIGINAL_IMAGE_DIR"]+"/*")
  df_test = pd.DataFrame(list_original_image_files, columns=["Original_Image_Path"])
  return df_test

def test_model(yaml_file_path):
  """
  This function is to test model. this will save output csv to given path
  """
  config_dict = get_configuration_dict(yaml_file_path)
  config_dict = data_dir_prep(config_dict)
  df_test = prep_test_df(config_dict)
  predict_points(df_test, config_dict)
  return 