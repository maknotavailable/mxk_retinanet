
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

import keras
from utils.eval import evaluate
from utils.eval_polyp import evaluate_polyp
from utils.polyp_utils import recall, precision, f1, f2
import numpy as np
from azureml.core import Run

# start an Azure ML run
run = Run.get_context()

class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(
        self,
        generator,
        data_dir,
        train_dir,
        val_dir,
        ## can these be changed via arguments?
        ## changed 
        iou_threshold=0.25,
        score_threshold=0.15,
        ## changed
        max_detections=500,
        save_path=None,
        tensorboard=None,
        weighted_average=False,
        verbose=1,
        dataset_type='ead'
    ):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.
        # Arguments
            generator        : The generator that represents the dataset to evaluate.
            iou_threshold    : The threshold used to consider when a detection is positive or negative.
            score_threshold  : The score confidence threshold to use for detections.
            max_detections   : The maximum number of detections to use per image.
            save_path        : The path to save images with visualized detections to.
            tensorboard      : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            weighted_average : Compute the mAP using the weighted average of precisions among classes.
            verbose          : Set the verbosity level, by default this is set to 1.
        """
        self.generator       = generator
        self.data_dir        = data_dir
        self.train_dir       = train_dir
        self.val_dir         = val_dir
        self.iou_threshold   = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections  = max_detections
        self.save_path       = save_path
        self.tensorboard     = tensorboard
        self.weighted_average = weighted_average
        self.verbose         = verbose
        self.dataset_type    = dataset_type

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # run evaluation
        
        if self.dataset_type == 'ead':
        
          false_positives_dict, true_positives_dict, iou_dict, average_precisions, iou, image_names, detection_list, scores_list, labels_list = evaluate(
              self.generator,
              self.model,
              iou_threshold=self.iou_threshold,
              score_threshold=self.score_threshold,
              max_detections=self.max_detections,
              save_path=self.save_path
          )

          # compute per class average precision
          total_instances = []
          precisions = []
          ious = []
          for label, (average_precision, num_annotations ) in average_precisions.items():
              if self.verbose == 1:
                  print('{:.0f} instances of class'.format(num_annotations),
                        self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision)
                        , 'and average IoU: {:.2f}'.format(iou_dict[label][0]))
              total_instances.append(num_annotations)
              precisions.append(average_precision)
              ious.append(iou_dict[label][0])
          if self.weighted_average:
              self.mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
          else:
              self.mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)

          #print(precisions)
          ## i think here tensorboard file is written
          if self.tensorboard is not None and self.tensorboard.writer is not None:
              import tensorflow as tf
              summary = tf.Summary()

              summary_value = summary.value.add()
              summary_value.simple_value = self.mean_ap
              summary_value.tag = "mAP"
              self.tensorboard.writer.add_summary(summary, epoch)
              run.log('mAP', self.mean_ap)

              self.mIoU = np.mean(iou)
              summary_value = summary.value.add()
              summary_value.simple_value = self.mIoU
              summary_value.tag = "mIoU"
              self.tensorboard.writer.add_summary(summary, epoch)
              run.log('mIoU', self.mIoU)

              self.EAD_Score_old = 0.8*self.mean_ap + 0.2*self.mIoU
              summary_value = summary.value.add()
              summary_value.simple_value = self.EAD_Score_old
              summary_value.tag = "EAD Score (old)"
              self.tensorboard.writer.add_summary(summary, epoch)    
              run.log('EAD_Score_old', self.EAD_Score_old)

              self.EAD_Score = 0.6*self.mean_ap + 0.4*self.mIoU
              summary_value = summary.value.add()
              summary_value.simple_value = self.EAD_Score
              summary_value.tag = "EAD Score"
              self.tensorboard.writer.add_summary(summary, epoch)    
              run.log('EAD_Score', self.EAD_Score)

              self.AP1 = precisions[0]
              total_instances.append(num_annotations)
              summary_value = summary.value.add()
              summary_value.simple_value = self.AP1
              summary_value.tag = "specularity mAP"
              self.tensorboard.writer.add_summary(summary, epoch)
              run.log('specularity mAP', self.AP1)

              self.IOU1 = ious[0]
              summary_value = summary.value.add()
              summary_value.simple_value = self.IOU1
              summary_value.tag = "specularity IOU"
              self.tensorboard.writer.add_summary(summary, epoch)
              run.log('specularity IOU', self.IOU1)

              self.AP2 = precisions[1]
              summary_value = summary.value.add()
              summary_value.simple_value = self.AP2
              summary_value.tag = "saturation mAP"
              self.tensorboard.writer.add_summary(summary, epoch)
              run.log('saturation mAP', self.AP2)

              self.IOU2 = ious[1]
              summary_value = summary.value.add()
              summary_value.simple_value = self.IOU2
              summary_value.tag = "saturation IOU"
              self.tensorboard.writer.add_summary(summary, epoch)
              run.log('saturation IOU', self.IOU2)

              self.AP3 = precisions[2]
              summary_value = summary.value.add()
              summary_value.simple_value = self.AP3
              summary_value.tag = "artifact mAP"
              self.tensorboard.writer.add_summary(summary, epoch)
              run.log('artifact mAP', self.AP3)

              self.IOU3 = ious[2]
              summary_value = summary.value.add()
              summary_value.simple_value = self.IOU3
              summary_value.tag = "artifact IOU"
              self.tensorboard.writer.add_summary(summary, epoch)
              run.log('artifact IOU', self.IOU3)   

              self.AP4 = precisions[3]
              summary_value = summary.value.add()
              summary_value.simple_value = self.AP4
              summary_value.tag = "blur mAP"
              self.tensorboard.writer.add_summary(summary, epoch)     
              run.log('blur mAP', self.AP4)  

              self.IOU4 = ious[3]
              summary_value = summary.value.add()
              summary_value.simple_value = self.IOU4
              summary_value.tag = "blur IOU"
              self.tensorboard.writer.add_summary(summary, epoch)
              run.log('blur IOU', self.IOU4)                  

              self.AP5 = precisions[4]
              summary_value = summary.value.add()
              summary_value.simple_value = self.AP5
              summary_value.tag = "contrast mAP"
              self.tensorboard.writer.add_summary(summary, epoch) 
              run.log('contrast mAP', self.AP5)

              self.IOU5 = ious[4]
              summary_value = summary.value.add()
              summary_value.simple_value = self.IOU5
              summary_value.tag = "contrast IOU"
              self.tensorboard.writer.add_summary(summary, epoch)
              run.log('contrast IOU', self.IOU5) 

              self.AP6 = precisions[5]
              summary_value = summary.value.add()
              summary_value.simple_value = self.AP6
              summary_value.tag = "bubbles mAP"
              self.tensorboard.writer.add_summary(summary, epoch) 
              run.log('bubbles mAP', self.AP6)

              self.IOU6 = ious[5]
              summary_value = summary.value.add()
              summary_value.simple_value = self.IOU6
              summary_value.tag = "bubbles IOU"
              self.tensorboard.writer.add_summary(summary, epoch)
              run.log('bubbles IOU', self.IOU6) 

              self.AP7 = precisions[6]
              summary_value = summary.value.add()
              summary_value.simple_value = self.AP7
              summary_value.tag = "instrument mAP"
              self.tensorboard.writer.add_summary(summary, epoch) 
              run.log('instrument mAP', self.AP7)

              self.IOU7 = ious[6]
              summary_value = summary.value.add()
              summary_value.simple_value = self.IOU7
              summary_value.tag = "instrument IOU"
              self.tensorboard.writer.add_summary(summary, epoch)
              run.log('instrument IOU', self.IOU7) 

          logs['mAP'] = self.mean_ap
          logs["mIoU"] = self.mIoU
          logs["EAD_Score_old"] = self.EAD_Score_old
          logs["EAD_Score"] = self.EAD_Score
          logs["specularity mAP"] = self.AP1
          logs["saturation mAP"] = self.AP2
          logs["artifact mAP"] = self.AP3
          logs["blur mAP"] = self.AP4
          logs["contrast mAP"] = self.AP5
          logs["bubbles mAP"] = self.AP6
          logs["instrument mAP"] = self.AP7
          logs["specularity mAP"] = self.IOU1
          logs["saturation mAP"] = self.IOU2
          logs["artifact mAP"] = self.IOU3
          logs["blur mAP"] = self.IOU4
          logs["contrast mAP"] = self.IOU5
          logs["bubbles mAP"] = self.IOU6
          logs["instrument mAP"] = self.IOU7

          ##


          if self.verbose == 1:
              #print("Gamma, alpha: ", )
              print('mAP: {:.4f}'.format(self.mean_ap))
              print('mIoU: {:.4f}'.format(self.mIoU))
              print('EAD Score (old): {:.4f}'.format(self.EAD_Score_old))
              print('EAD Score: {:.4f}'.format(self.EAD_Score))
            
            
        elif self.dataset_type == 'polyp':
          
          self.TP, self.FP, self.TN, self.FN, p_count = evaluate_polyp(
              self.generator,
              self.model,
              self.data_dir,
              self.val_dir,
              iou_threshold=self.iou_threshold,
              score_threshold=self.score_threshold,
              max_detections=self.max_detections,
              save_path=self.save_path
          )

          self.precision   = precision(self.TP,self.FP)
          self.recall      = recall(self.TP,self.FN)
          self.specificity = specificity(self.TN, self.FP, self.FN)
          self.f1          = f1(self.precision, self.recall)
          self.f2          = f2(self.precision, self.recall)

          if self.tensorboard is not None and self.tensorboard.writer is not None:
              import tensorflow as tf
              summary = tf.Summary()

              summary_value = summary.value.add()
              summary_value.simple_value = self.TP
              summary_value.tag = "TP"
              self.tensorboard.writer.add_summary(summary, epoch)
              #run.log('TP', self.TP)

              summary_value = summary.value.add()
              summary_value.simple_value = self.FP
              summary_value.tag = "FP"
              self.tensorboard.writer.add_summary(summary, epoch)
              #run.log('FP', self.FP)

              summary_value = summary.value.add()
              summary_value.simple_value = self.TN
              summary_value.tag = "TN"
              self.tensorboard.writer.add_summary(summary, epoch)
              #run.log('TN', self.TN) 

              summary_value = summary.value.add()
              summary_value.simple_value = self.FN
              summary_value.tag = "FN"
              self.tensorboard.writer.add_summary(summary, epoch)
              #run.log('FN', self.FN) 

              summary_value = summary.value.add()
              summary_value.simple_value = self.precision
              summary_value.tag = "precision"
              self.tensorboard.writer.add_summary(summary, epoch)
              #run.log('precision', self.precision) 

              summary_value = summary.value.add()
              summary_value.simple_value = self.recall
              summary_value.tag = "recall"
              self.tensorboard.writer.add_summary(summary, epoch)
              #run.log('recall', self.recall)

              summary_value = summary.value.add()
              summary_value.simple_value = self.f1
              summary_value.tag = "f1"
              self.tensorboard.writer.add_summary(summary, epoch)
              #run.log('f1', self.f1)

              summary_value = summary.value.add()
              summary_value.simple_value = self.f2
              summary_value.tag = "f2"
              self.tensorboard.writer.add_summary(summary, epoch)
              #run.log('f2', self.f2)

          logs['TP'] = self.TP
          logs["FP"] = self.FP
          logs["TN"] = self.TN
          logs["FN"] = self.FN
          logs['precision'] = self.precision
          logs["recall"] = self.recall
          logs["f1"] = self.f1
          logs["f2"] = self.f2

          if self.verbose == 1:
              print("TP: %d\nFP: %d\nFN: %d\nNumber of Polyps: %d\n"%(self.TP, self.FP, self.FN, p_count))
              print('precision: {:.4f}'.format(self.precision))
              print('recall: {:.4f}'.format(self.recall))
              print('f1: {:.4f}'.format(self.f1))
              print('f2: {:.4f}'.format(self.f2))

              
         