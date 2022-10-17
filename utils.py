import cv2
import numpy as np
def image_enhancement(image):
  """
  This function take the image and enhance it with various ways

  parameters:
      -image: numpy array of image

  return:
      -image: numpy array of image
  """
  min = np.amin(image)
  max = np.amax(image)
  enhanced_image = (255/(max-min))*(image-min)
  return enhanced_image.astype("uint8")


def euclidean_distance(point1, point2):
  """
  parameters:
      -point1: numpy array having x,y
      -point2: numpy array having x,y
    
  return:
      -distance: float, distance between input points
  """
  try:

    distance = (np.sum((point1-point2)**2))**.5
    return distance
  except:
    print("Problem in calculating euclidean distance. Check the point numpy arrays.")


def hamming_distance(n1, n2):
  """
  This function gives Hamming deistance between two given numbers

  parameters:
      -n1, n2: two integers

  Return:
      -setBits: Hamming distance  
  """
  x = n1 ^ n2
  setBits = 0
  while (x > 0) :
      setBits += x & 1
      x >>= 1
  return setBits


def list_hamming_distance(list1, list2):
  """
  This function gives hamming distance between 2 vectors of integers

  parameters:
      -list1, list2: lists of integers

  returns:
      -ham_dist: Hamming distance between lists
  """
  zip_list = zip(list1, list2)
  ham_dist=0
  for l1, l2 in zip_list:
    ham_dist += hamming_distance(l1, l2)
  return ham_dist