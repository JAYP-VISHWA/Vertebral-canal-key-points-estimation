
from data_prep import data_dir_prep
from reference_feat import prep_reference_df, reference_feature_extraction
from utils import image_enhancement, euclidean_distance


import numpy as np
import cv2
import pandas as pd

def best_predicted_point(reference_feature, predicted_keypoints, predicted_descriptors, config_dict):
  """
  This function select best predicted points for given feature of a single reference point based on  distance threshold

  parameters:
      -reference_feature: numpy array with single reference feature
      -predicted_keypoints: numpy array
      -predicted_descriptors: numpy array
      -config_dict: dictionary with configuration

  returns:
      -selected_points: seleced keypoints array
      -selected_descriptors: descriptors corresponding to selected points
      -selected_distances: distance corresponding to selected points
      -unselected_points: rejected points array
      -unselected_descriptors: descriptors corresponding to rejected points
      -unselected_distances: distance corresponding to rejected points
  """
  selected_points = []
  selected_descriptors = []
  selected_distances = []

  unselected_points = []
  unselected_descriptors = []
  unselected_distances = []
  zip_point_des = zip(predicted_keypoints, predicted_descriptors)
  for point, descriptor in zip_point_des:
    distance = euclidean_distance(reference_feature, descriptor)
    if distance < config_dict["EUCLIDEAN_DISTANCE_MAX_THRESHOLD"]: # and distance>config_dict["EUCLIDEAN_DISTANCE_MIN_THRESHOLD"]:  #min_distance:
      selected_points.append(point)
      selected_descriptors.append(descriptor)
      selected_distances.append(distance)
    else:
      unselected_points.append(point)
      unselected_descriptors.append(descriptor)
      unselected_distances.append(distance)

  return selected_points, selected_descriptors, selected_distances, unselected_points, unselected_descriptors, unselected_distances



def point_selection_and_ref(reference_features, predicted_keypoints, predicted_descriptors, config_dict):
  """
  This function gives keypoints whose descriptors with in threshold from all reference descriptors

  parameters:
      -reference_features: Array of all reference features
      -predicted_keypoints: predicted keypoints array
      -predicted_descriptors: descriptors corresponding to predicted keypoints
      -config_dict: dictionary with configuration

  returns:
      -all_selected_points: All the points which satisfy the threshold from all the reference descriptors 
      -all_selected_descriptors: corresponding descriptors
      -all_selected_distances: corresponding distances
  """
  is_first_instance = True
  for feat in reference_features:
    # min_descriptor, min_point, min_distance = hamming_best_point(feat, predicted_keypoints, predicted_descriptors, config_dict)
    if is_first_instance:
      selected_points, selected_descriptors, selected_distances,_,_,_ = best_predicted_point(feat, predicted_keypoints, predicted_descriptors, config_dict)
      is_first_instance = False
    else:
      selected_points, selected_descriptors, selected_distances,_,_,_ = best_predicted_point(feat, selected_points, selected_descriptors, config_dict)
  return selected_points, selected_descriptors, selected_distances


def point_selection_or_ref(reference_features, predicted_keypoints, predicted_descriptors, config_dict):
  """
  This function gives keypoints whose descriptors with in threshold from any one reference descriptors

  parameters:
      -reference_features: Array of all reference features
      -predicted_keypoints: predicted keypoints array
      -predicted_descriptors: descriptors corresponding to predicted keypoints
      -config_dict: dictionary with configuration

  returns:
      -all_selected_points: All the points which satisfy the threshold from any one of the reference descriptors 
      -all_selected_descriptors: corresponding descriptors
      -all_selected_distances: corresponding distances 
  """
  pred_des = predicted_descriptors
  pred_keyp = predicted_keypoints
  all_selected_points = []
  all_selected_descriptors = []
  all_selected_distances = [] 
  for feat in reference_features:
    selected_points, selected_descriptors, selected_distances, pred_keyp, pred_des,_ = best_predicted_point(feat, pred_keyp, pred_des, config_dict)
    all_selected_points += selected_points
    all_selected_descriptors += selected_descriptors
    all_selected_distances += selected_distances
  return all_selected_points, all_selected_descriptors, all_selected_distances

def point_pairs_selection(start_points, start_distances, end_points, end_distances, config_dict):
  """
  This function gives pairs of extreme points of canal based on distance threshold between points. here points are point objects given by cv2 SIFT

  parameters:
      -start_points: array of start point (one of the canal extreme point)
      -start_distances: array of corresponding distances
      -end_points: array of end points (another canal extreme point)
      -end_distances: array of distances corresponding to end points
      -config_dict: configuration dictionary

  returns:
      -selected_pairs: all pairs of point objects which satisfied the threshold
  """
  selected_pairs = []
  for s_point in start_points:
    for e_point in end_points:
      euclid_distance = euclidean_distance(np.array(s_point.pt), np.array(e_point.pt))
      # print("Eu distance: ", euclid_distance)
      if euclid_distance<config_dict["PAIR_DISTANCE_MAX_THRESHOLD"] and euclid_distance>config_dict["PAIR_DISTANCE_MIN_THRESHOLD"]:
        selected_pairs.append([s_point, e_point])
  return selected_pairs

def pair_alignment_filter(point_pair, config_dict):
  """
  This funtions checks alignment of point pairs

  parameters:
      -point_pair: list of 2 point objects
      -config_dict: configuration dictionary

  return:
  True or False based on comparison with threshold
  """
  s_point = point_pair[0].pt
  e_point = point_pair[1].pt
  if np.abs(s_point[0]-e_point[0])<config_dict["PAIR_ALIGNMENT_THRESHOLD"]:
    return True
  # elif np.abs(s_point[1]-e_point[1])<config_dict["PAIR_ALIGNMENT_THRESHOLD"]:
  #   return True
  else:
    return False


def predict_points(df_ref, config_dict):
  """
  This function gives final prediction of points

  parameters:
      -df_ref: dataframe having paths of image files
      -config_dict: configuration dictionary
  """
  feature_extractor = cv2.SIFT_create()
  np_st = np.load(config_dict["REFERENCE_START_POINT_FEATURE_PATH"])
  np_en = np.load(config_dict["REFERENCE_END_POINT_FEATURE_PATH"])
  df_output = pd.DataFrame(columns=["Image_file_path", "Length_of_Canal"])
  for row in df_ref.iterrows():
    img = cv2.imread(row[1][0], cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = feature_extractor.detectAndCompute(img, None)

    start_points, _, start_distances = point_selection_or_ref(np_st, keypoints, descriptors, config_dict)
    end_points, _, end_distances = point_selection_or_ref(np_en, keypoints, descriptors, config_dict)
    # print("Number of start points: ", len(start_points))
    # print("Number of end points: ", len(end_points))
    selected_pairs = point_pairs_selection(start_points, start_distances, end_points, end_distances, config_dict)
    final_pair_num = 0
    final_pairs_distance = []
    for point_pair in selected_pairs:
      if pair_alignment_filter(point_pair, config_dict):
        final_pair_num += 1
        final_pairs_distance.append(euclidean_distance(np.array(point_pair[0].pt), np.array(point_pair[1].pt)))
        img = cv2.circle(img, [round(_) for _ in point_pair[0].pt], 1, (250,0,0), 2)
        img = cv2.circle(img, [round(_) for _ in point_pair[1].pt], 1, (250,0,0), 2)
    df_output.loc[len(df_output.index)] = [row[1][0], np.round(np.mean(final_pairs_distance),2)]
  df_output.to_csv(config_dict["OUTPUT_FILE_PATH"])
    