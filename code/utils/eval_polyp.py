## evaluate_polyp

from utils.compute_overlap import compute_overlap
from utils.visualization import draw_detections, draw_annotations

import keras
import numpy as np
import os
import time

import cv2
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."

from utils.polyp_utils import run_validation, df_builder

def _get_detections_p(generator, model, mode, score_threshold=0.05, max_detections=100, save_path=None, im_threshold=0.3):
    """ Get the detections from the model using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]
    
    ## added by me
    image_names    = []
    detection_list = []
    scores_list    = []
    labels_list    = []
    duration_list  = []
    
    for i in range(generator.size()): #progressbar.progressbar(, prefix='Running network: '):
        raw_image    = generator.load_image(i)
        ## i added the names part
        image_name   = generator.image_path(i)
        image_names.append(image_name)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)
        
        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        start    = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]
        duration = time.time() - start
        duration_list.append(duration)

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]


        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        ## annotations for drawing:
        detection_list.append(image_boxes)
        image_scores     = scores[scores_sort]
        scores_list.append(image_scores)
        image_labels     = labels[0, indices[scores_sort]]
        labels_list.append(image_labels)
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
        
        if save_path is not None:
            ## both annotations and detections are drawn an "raw_image"
            draw_annotations(raw_image, generator.load_annotations(i), mode, label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, mode, label_to_name=generator.label_to_name, im_threshold=im_threshold)
            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

    #print("scores_list: ",scores_list)
    #print("labels_list: ",labels_list)
    avg_time = np.mean(duration_list)
    print("avg time for inference per image: ",avg_time)
    return all_detections, image_names, detection_list, scores_list, labels_list


def evaluate_polyp(
    generator,
    model,
    data_dir,
    val_dir,
    val_annotations,
    mode,
    classes,
    iou_threshold=0.25,
    score_threshold=0.05,
    max_detections=500,
    save_path=None,
    im_threshold=0.3
):
    
    all_detections, image_names, detection_list, scores_list, labels_list     = _get_detections_p(generator, model, mode, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path, im_threshold=im_threshold)
    # build df
    detections_df  = df_builder(scores_list, image_names, detection_list, labels_list, mode, classes)
    print("# of polyp detections: ",detections_df[detections_df["object_id"] == "polyp"].shape[0])
    print("# of artefact detections: ",detections_df[detections_df["object_id"] != "polyp"].shape[0])
    a_count = detections_df[detections_df["object_id"] != "polyp"].shape[0]
    if mode == "scoring":
      mask_dir = os.path.join(data_dir,val_dir)
      annot_csv = os.path.join(data_dir,os.path.join(data_dir,val_annotations))
      TP, FP, TN, FN, p_count = run_validation(detections_df, annot_csv, mask_dir)
      return TP, FP, TN, FN, p_count, a_count
    elif mode == "detection":
      return detections_df
    else:
      print("ERROR: non-valid mode argument")
      return