# takes as input the dataframe with all the detections in submission-style (EAD) format
import time
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def center_based_validation(per_im_df, mask_source):

  # load the mask
  mask = np.array(Image.open(mask_source), dtype=np.uint8)
  
  # initialize all counters
  true_positives  = 0
  false_positives = 0
  false_negatives = 0
  num_detections = per_im_df.shape[0]
  
  # get the individual components
  gt_components  = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
  components_mat = gt_components[1].astype(np.uint8)
  num_components = gt_components[0] - 1 
  
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


def run_validation(df, mask_dir):
  
  start = time.time()
  
  df["c_x"] = df["x1"] + ((df["x2"]-df["x1"])/2).astype(int)
  df["c_y"] = df["y1"] + ((df["y2"]-df["y1"])/2).astype(int)
  
  image_list = list(df["image_path"].unique())

  TP_overall = 0
  FP_overall = 0
  TN_overall = 0
  FN_overall = 0
  tot_polyps = 0
  
  for img in image_list:
    per_im_df   = df[df["image_path"] == img]
    
    mask_source = img.replace(mask_dir.replace("_masks",""),mask_dir)
    if "ETIS" in mask_dir:
      mask_source = mask_source.replace("_masks/","_masks/p")
    TP_im, FP_im, TN_im, FN_im, num = center_based_validation(per_im_df, mask_source)

    TP_overall += TP_im
    FP_overall += FP_im
    TN_overall += TN_im
    FN_overall += FN_im
    tot_polyps += num
    
  print('Time required for validation (in seconds): %.0f'%(time.time() - start))
  return TP_overall, FP_overall, TN_overall, FN_overall, tot_polyps

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