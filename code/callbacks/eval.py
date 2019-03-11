
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
        ## can these be changed via arguments?
        ## changed 
        iou_threshold=0.25,
        score_threshold=0.15,
        ## changed
        max_detections=500,
        save_path=None,
        tensorboard=None,
        weighted_average=False,
        verbose=1
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
        self.iou_threshold   = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections  = max_detections
        self.save_path       = save_path
        self.tensorboard     = tensorboard
        self.weighted_average = weighted_average
        self.verbose         = verbose

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # run evaluation
        false_positives_dict, true_positives_dict, average_precisions, iou, image_names, detection_list, scores_list, labels_list = evaluate(
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
        for label, (average_precision, num_annotations ) in average_precisions.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)
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
            
            self.AP2 = precisions[1]
            summary_value = summary.value.add()
            summary_value.simple_value = self.AP2
            summary_value.tag = "saturation mAP"
            self.tensorboard.writer.add_summary(summary, epoch)
            run.log('saturation mAP', self.AP2)
            
            self.AP3 = precisions[2]
            summary_value = summary.value.add()
            summary_value.simple_value = self.AP3
            summary_value.tag = "artifact mAP"
            self.tensorboard.writer.add_summary(summary, epoch)
            run.log('artifact mAP', self.AP3)
            
            self.AP4 = precisions[3]
            summary_value = summary.value.add()
            summary_value.simple_value = self.AP4
            summary_value.tag = "blur mAP"
            self.tensorboard.writer.add_summary(summary, epoch)     
            run.log('blur mAP', self.AP4)       

            self.AP5 = precisions[4]
            summary_value = summary.value.add()
            summary_value.simple_value = self.AP5
            summary_value.tag = "contrast mAP"
            self.tensorboard.writer.add_summary(summary, epoch) 
            run.log('contrast mAP', self.AP5)
            
            self.AP6 = precisions[5]
            summary_value = summary.value.add()
            summary_value.simple_value = self.AP6
            summary_value.tag = "bubbles mAP"
            self.tensorboard.writer.add_summary(summary, epoch) 
            run.log('bubbles mAP', self.AP6)
            
            self.AP7 = precisions[6]
            summary_value = summary.value.add()
            summary_value.simple_value = self.AP7
            summary_value.tag = "instrument mAP"
            self.tensorboard.writer.add_summary(summary, epoch) 
            run.log('instrument mAP', self.AP7)
            
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
        
        ##
        

        if self.verbose == 1:
            #print("Gamma, alpha: ", )
            print('mAP: {:.4f}'.format(self.mean_ap))
            print('mIoU: {:.4f}'.format(self.mIoU))
            print('EAD Score (old): {:.4f}'.format(self.EAD_Score_old))
            print('EAD Score: {:.4f}'.format(self.EAD_Score))