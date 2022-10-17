from yaml_reader import get_configuration_dict
from data_prep import data_dir_prep
from reference_feat import prep_reference_df, reference_feature_extraction
from utils import image_enhancement, euclidean_distance
from point_selection import *
from canal import *

yaml_file_path = "synap.yaml"
create_reference_files(yaml_file_path)
test_model(yaml_file_path)