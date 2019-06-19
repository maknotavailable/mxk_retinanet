import time
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np

def get_components(mask):
    
    gt_components  = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    components_mat = gt_components[1].astype(np.uint8)
    num_components = gt_components[0] - 1 
    
    # check for noise if there is more than one polyp
    if num_components > 1:
        it_length = num_components
        for i in range(it_length):
            current_component = np.zeros(shape=components_mat.shape, dtype=np.uint8)
            current_component[components_mat == i+1] = 255
            pixel_count = np.count_nonzero(current_component == 255)
            if pixel_count < 10:
                num_components -= 1
    
    return num_components, components_mat
  

def no_det_on_image(mask_source):
  
  # load the mask
  mask = np.array(Image.open(mask_source), dtype=np.uint8)
  
  # get the individual components
  num_components, components_mat = get_components(mask)

  true_positives  = 0
  false_positives = 0
  false_negatives = num_components
  
  # TN
  if num_components == 0:
    true_negatives = 1
  else:
    true_negatives = 0

  return true_positives, false_positives, true_negatives, false_negatives, num_components


def center_based_validation(per_im_df, mask_source):

  # load the mask
  mask = np.array(Image.open(mask_source), dtype=np.uint8)
  
  # initialize all counters
  true_positives  = 0
  false_positives = 0
  false_negatives = 0
  num_detections  = per_im_df.shape[0]
  
  # get the individual components
  num_components, components_mat = get_components(mask)
  
  # iterate through each box
  for box in range(num_detections):

    # get the center coordinates
    x = per_im_df["c_x"].iloc[box]
    y = per_im_df["c_y"].iloc[box]

    # TP and FP
    # note that PIL IMAGE works with x and y the other way around
    # if center is not a 0 pixel than we have a GT
    if mask[y,x] > 0:
      true_positives += 1
      # finding component
      comp = components_mat[y,x]
      # removing ground-truth for future check-ups
      mask[components_mat == comp] = 0
    else:
      false_positives += 1
      
  # FN
  false_negatives = num_components - true_positives

  # TN
  if num_components == 0 and num_detections == 0:
    true_negatives = 1
  else:
    true_negatives = 0

  return true_positives, false_positives, true_negatives, false_negatives, num_components


def run_validation(df, annot_csv, mask_dir):
    
  df["c_x"] = df["x1"] + ((df["x2"]-df["x1"])/2).astype(int)
  df["c_y"] = df["y1"] + ((df["y2"]-df["y1"])/2).astype(int)
  
  image_list  = list(df["image_path"].unique())
  annot_df    = pd.read_csv(annot_csv, names=["image_path", "x1", "y1", "x2", "y2", "object_id"])
  annot_list  = list(annot_df["image_path"].unique())
  no_det_list = list(set(annot_list) - set(image_list))

  TP_overall = 0
  FP_overall = 0
  TN_overall = 0
  FN_overall = 0
  tot_polyps = 0
  
  polyp_df = df[df["object_id"] == "polyp"]
  
  for img in image_list:
    per_im_df   = polyp_df[polyp_df["image_path"] == img]
    if "combined" in mask_dir:
      mask_dir = mask_dir.replace("combined","polyps")
    mask_source = img.replace(mask_dir.replace("_masks",""),mask_dir)
    if "612" in mask_dir:
      mask_source = mask_source.replace(".bmp",".tif")
    if "ETIS" in mask_dir:
      mask_source = mask_source.replace("_masks/","_masks/p")
    TP_im, FP_im, TN_im, FN_im, num = center_based_validation(per_im_df, mask_source)

    TP_overall += TP_im
    FP_overall += FP_im
    TN_overall += TN_im
    FN_overall += FN_im
    tot_polyps += num
    
  for img in no_det_list:
    if "combined" in mask_dir:
      mask_dir = mask_dir.replace("combined","polyps")
    mask_source = img.replace(mask_dir.replace("_masks",""),mask_dir)
    if "612" in mask_dir:
      mask_source = mask_source.replace(".bmp",".tif")
    if "ETIS" in mask_dir:
      mask_source = mask_source.replace("_masks/","_masks/p")
    TP_im, FP_im, TN_im, FN_im, num = no_det_on_image(mask_source)
    
    TP_overall += TP_im
    FP_overall += FP_im
    TN_overall += TN_im
    FN_overall += FN_im
    tot_polyps += num
    
  return TP_overall, FP_overall, TN_overall, FN_overall, tot_polyps

def df_builder(scores_list, image_names, detection_list, labels_list, mode, classes):
  
  class_file = pd.read_csv(classes, names=["class_name","label"])
  class_list  = list(class_file.class_name)
    
  #start = time.time()
  col_names =  ["image_path", "x1", "y1", "x2", "y2", "object_id", "score"]
  detections_df = pd.DataFrame(columns = col_names)
  
  for i in range(len(scores_list)):

    ## to change
    image_path = image_names[i]
    detections = detection_list[i]
    scores = scores_list[i]
    labels = labels_list[i]
    
    if mode == "scoring":
      for j in range(len(scores)):
        object_id = class_list[labels[j]]
        x1 = int(detections[j][0])
        y1 = int(detections[j][1])
        x2 = int(detections[j][2])
        y2 = int(detections[j][3])

        score = scores[j]

        df_row = pd.DataFrame([[image_path, x1, y1, x2, y2, object_id, score]], columns = col_names)
        detections_df = detections_df.append(df_row, ignore_index=True)
        
    elif mode == "detection":
      for j in range(len(scores)):
        #class_labels = ["specularity", "saturation", "artifact", "blur", "contrast", "bubbles", "instrument"]
        object_id = class_list[labels[j]]
        x1 = int(detections[j][0])
        y1 = int(detections[j][1])
        x2 = int(detections[j][2])
        y2 = int(detections[j][3])

        score = scores[j]

        df_row = pd.DataFrame([[image_path, x1, y1, x2, y2, object_id, score]], columns = col_names)
        detections_df = detections_df.append(df_row, ignore_index=True)      

  
  #print('Duration for building df: %.0f'%(time.time() - start))  
  return detections_df

# metrics
def precision(TP, FP):
  if (TP+FP) != 0:
    return TP/(TP+FP)
  else:
    return 0

def recall(TP,FN):
  if (TP+FN) != 0:
    return TP/(TP+FN)
  else:
    return 0
  
def f1(prec, rec):
  if (prec+rec) != 0:  
    return (2*prec*rec)/(prec+rec)
  else:
    return 0
  
def f2(prec, rec):
  if (4*prec+rec) != 0: 
    return (5*prec*rec)/(4*prec+rec)
  else:
    return 0