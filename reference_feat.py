import glob
import cv2
import numpy as np
import pandas as pd

from utils import image_enhancement, euclidean_distance

def prep_reference_df(config_dict):
  """
  This function prepares reference dataframe
  
  parameters:
      -config_dict: configuration dictionary

  return:
      -df_reference: dataframe with original and tagged file paths and corresponding reference start and end points

  """
  list_original_image_files = glob.glob(config_dict["ORIGINAL_IMAGE_DIR"]+"/*")
  list_tagged_image_files = glob.glob(config_dict["TAGGED_IMAGE_DIR"]+"/*")
  df_reference = pd.DataFrame(columns=["Original_Image_Path", "Tagged_Image_Path", "Start_Point", "End_Point"])
  zip_image_path_files = zip(list_tagged_image_files, list_original_image_files)
  for image_file, org_image_path in zip_image_path_files: #list_tagged_image_files:
    img = cv2.imread(image_file)
    bw_img = (img[:,:,0]>250) * (img[:,:,1]<20) 
    points = np.where(bw_img==1)
    start_point = (points[1][0], points[0][0])
    end_point = (points[1][-1], points[0][-1])
    # list_name = image_file.split("/")
    # list_name[3] = "original_images"
    # list_name[-1] = list_name[-1].split("_")[-1]
    # org_image_path = "/".join(list_name)
    df_reference.loc[len(df_reference.index)] = [org_image_path, image_file, start_point, end_point]
  return df_reference



def reference_feature_extraction(config_dict, df_reference):
  """
  This function extract and saves reference SIFT features

  parameters:
      -df_reference: reference dataframe with original and tagged file paths with corresponding refereence start and end points

  output:
      -start_point_features.npy: file having start point features
      -end_point_features.npy: file having end point features
  """
  try:
    np_features_start = []
    np_features_end = []
    feature_extractor = cv2.SIFT_create() 
    # nfeatures = 1,
    #                                     nOctaveLayers = 3,
    #                                     contrastThreshold = 0.04,
    #                                     edgeThreshold = 10,
    #                                     sigma = 1.6 )    #(nfeatures=1500)

    for row in df_reference.iterrows():
        img = cv2.imread(row[1][0], cv2.IMREAD_GRAYSCALE)
        img = image_enhancement(img)
        # # img.reshape(img.shape[0], img.shape[1])
        keypoints, descriptors = feature_extractor.detectAndCompute(img, None)
        reference_start_point = row[1][2]
        reference_end_point = row[1][3]
        list_dist_start = []
        list_dist_end = []
        point_list = []

        for point in keypoints:
          point_list.append(point.pt)
          list_dist_start.append(euclidean_distance(np.array(reference_start_point), np.array(point.pt)))
          list_dist_end.append(euclidean_distance(np.array(reference_end_point), np.array(point.pt)))
        min_dist_start = min(list_dist_start)
        min_dist_end = min(list_dist_end)
        ind_start = list_dist_start.index(min_dist_start)
        ind_end = list_dist_end.index(min_dist_end)

        if min_dist_start<5 and min_dist_end<5:
          np_features_start.append(descriptors[ind_start])
          np_features_end.append(descriptors[ind_end])
        np.save(config_dict["REFERENCE_FEATURE_PATH"]+"/start_point_features.npy", np.array(np_features_start))
        np.save(config_dict["REFERENCE_FEATURE_PATH"]+"/end_point_features.npy", np.array(np_features_end))
    return

  except:
    print("Error in feature extraction. Look at reference_feature_extraction.")
