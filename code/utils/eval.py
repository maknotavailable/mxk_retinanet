"""
Copyright 2017-2018 Fizyr (https://fizyr.com)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from compute_overlap import compute_overlap
from utils.visualization import draw_detections, draw_annotations

import keras
import numpy as np
import os

import cv2
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."

## this function is from EAD and not fizyr
def _compute_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
        mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec = rec.tolist()
    prec = prec.tolist()
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
      (goes from the end to the beginning)
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #   range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #   range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
      mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
    """
    # matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    i_list = []
    for i in range(1, len(mrec)):
      if mrec[i] != mrec[i-1]:
        i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
      (numerical integration)
    """
    # matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    ap = 0.0
    for i in i_list:
      ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap


def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
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
    image_names = []
    detection_list = []
    scores_list = []
    labels_list = []
    
    
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
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

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
            draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

    #print("scores_list: ",scores_list)
    #print("labels_list: ",labels_list)
    return all_detections, image_names, detection_list, scores_list, labels_list


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()

    return all_annotations


def evaluate(
    generator,
    model,
    iou_threshold=0.25,
    score_threshold=0.05,
    max_detections=500,
    save_path=None
):
    """ Evaluate a given dataset using a given model.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    ## all detections include all those that don't reach the treshold
    all_detections, image_names, detection_list, scores_list, labels_list     = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations    = _get_annotations(generator)
    #print(image_names)
    #print(detection_list)
    ## average_precisions is initialized as dictionary
    average_precisions = {}
    true_positives_dict = {}
    false_positives_dict = {}
    iou_dict = {}

    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))
    
    ## create different lists that I need
    iou = []
    #all_completed_detections = []
    #image_index = []
    #object_type_df = []

    # process detections and annotations
    ## all this part is done for each class
    ## but only for classes that are actually present
    ## so it doesn't cover classes that detections were made on but which were not in the picture
    ## however, generator.num_classes()=7
    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0
        num_detections = 0.0
        iou_per_class = []
        
        for i in range(generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []
            

            for d in detections:
                    
                scores = np.append(scores, d[4])
                
                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]
                
                
                ## here we check if box if it is true or false positive
                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    ## I assume that IoU is only calculated for TP boxes
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    #print("bbox overlap: ",max_overlap)
                    iou.append(np.asscalar(max_overlap))
                    iou_per_class.append(np.asscalar(max_overlap))
                    #print("iou list: ",iou)
                    detected_annotations.append(assigned_annotation)
                    #all_completed_detections.append(d)
                    #image_index.append(i)
                    #print(image_index)
                    num_detections += 1
                    #print(detections)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    ## the way (I think) they do it in EAD: if the overlap doesn't reach treshold, it's 0
                    iou.append(0)
                    iou_per_class.append(0)
            #print("Scores: ",scores)
                    
        # no annotations -> AP for this class is 0 (is this correct?)
        ## hid continue to see # of FP
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            #continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        ## cumsum returns the cumulative sum of the elements along a given axis
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        ## we divide an array by a scalar
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        
        ## average_precision will be in the following dictionary format: {0: (0.31672005587085506, 1112.0), 1: (0.08074755526818446, 107.0), 2: (0.19361940213603135, 291.0), 3: (0.12725467367537643, 57.0), 4: (0.20030495872509274, 121.0), 5: (0.06083609353943108, 481.0), 6: (0.41498412085028863, 89.0)}
        average_precisions[label] = average_precision, num_annotations
        ## added dictionaries for TP and FP
        ## I use max, because true_positives is an array with accumulating TP's
        if true_positives.size != 0:
            true_positives_dict[label] = max(true_positives), num_annotations
        else:
            true_positives_dict[label] = 0, num_annotations
        if false_positives.size != 0:
            false_positives_dict[label] = max(false_positives), num_annotations
        else:
            false_positives_dict[label] = 0, num_annotations
        iou_dict[label] = np.mean(iou), num_annotations
        #print("Label: ",generator.label_to_name(label))
        #print("FP: ",false_positives_dict)
        #print("TP: ",true_positives_dict)
        #print("AP: ", average_precision)
        #print("precision: ",precision)
        #print("recall: ",recall)
        

    return false_positives_dict, true_positives_dict, iou_dict, average_precisions, iou, image_names, detection_list, scores_list, labels_list