#!/usr/bin/env python
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


import argparse
import os
import sys
import pandas as pd
sys.path.append('./layers')
sys.path.append('./backend')
sys.path.append('./models')
sys.path.append('./preprocessing')
sys.path.append('./utils')
sys.path.append('./keras_resnet')

#from azureml.core import Run
#run = Run.get_context()

import warnings

import keras
import keras.preprocessing.image
import tensorflow as tf


# Change these to absolute imports if you copy this script outside the keras_retinanet package.
import layers  # noqa: F401
import losses
import models
from callbacks import RedirectModel
from callbacks.eval import Evaluate
from models.retinanet import retinanet_bbox
from preprocessing.csv_generator import CSVGenerator
from preprocessing.kitti import KittiGenerator
from preprocessing.open_images import OpenImagesGenerator
from preprocessing.pascal_voc import PascalVocGenerator
from utils.anchors import make_shapes_callback
from utils.config import read_config_file, parse_anchor_parameters
from utils.keras_version import check_keras_version
from utils.model import freeze as freeze_model
from utils.transform import random_transform_generator


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.
    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(fl_gamma, fl_alpha, r_weight, c_weight, p_weight, train_type, sample_t, backbone_retinanet, num_classes, weights, class_weights, loss_weights, multi_gpu=0,
                  freeze_backbone=False, lr=1e-5, config=None):
    """ Creates three models (model, training_model, prediction_model).
    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.
    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """

    modifier = freeze_model if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors   = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors   = anchor_params.num_anchors()

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model          = model_with_weights(backbone_retinanet(num_classes, train_type, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = model

    # make prediction model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

    if train_type == "single":
    # compile model
    
      training_model.compile(
          loss={
              'regression'    : losses.smooth_l1(r_weight),
              ## HERE THE INPUT ARGUMENTS CAN BE GIVEN: default focal(alpha=0.25, gamma=2.0)
              ## gamma: the actual "focusing parameter"
              'classification': losses.focal(c_weight, fl_alpha, fl_gamma, class_weights)
          },
          optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.001),
          #sample_weight_mode="temporal",
          #sample_weights=sample_t
      )
      
    elif train_type == "multi":
      training_model.compile(
          loss={
              'regression'    : losses.smooth_l1(r_weight),
              ## HERE THE INPUT ARGUMENTS CAN BE GIVEN: default focal(alpha=0.25, gamma=2.0)
              ## gamma: the actual "focusing parameter"
              'classification_artefact': losses.focal(c_weight, fl_alpha, fl_gamma),
              'classification_polyp': losses.focal2(p_weight, fl_alpha, fl_gamma)
          },
          optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.001)
          #loss_weights=loss_weights
      )      

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, args):
    """ Creates the callbacks to use during training.
    Args
        model: The base model.
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.
    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = args.tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = False,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)


    if args.evaluation and validation_generator:
        if args.dataset_type == 'coco':
            from callbacks.coco import CocoEval

            # use prediction model for evaluation
            evaluation = CocoEval(validation_generator, tensorboard=tensorboard_callback)
        else:
            evaluation = Evaluate(validation_generator, data_dir=args.data_dir, train_dir=args.train_dir, val_dir=args.val_dir, val_annotations=args.val_annotations, classes=args.classes, score_threshold=args.score_threshold, tensorboard=tensorboard_callback, weighted_average=args.weighted_average, dataset_type=args.dataset_type, mode=args.mode)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)
        

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(os.path.join(args.data_dir, args.snapshot_path))
        ## keras.callbacks.ModelCheckpoint: save model after every epoch
          
        if args.val_annotations:
            if args.dataset_type == 'ead':
              checkpoint = keras.callbacks.ModelCheckpoint(
                  os.path.join(
                      args.snapshot_path,
                      '{backbone}_{dataset_type}_p{prev_epoch}_{{epoch:02d}}_{{EAD_Score:.2f}}.h5'.format(backbone=args.backbone, dataset_type=args.dataset_type, prev_epoch=args.previous_epoch)
                  ),
                  ## I'm adding these things to always save a model (and overwrite) if it improves the score
                  verbose=1,
                  save_best_only=False,
                  monitor="EAD_Score",
                  mode='max'
              )
            elif args.dataset_type == 'polyp':
              checkpoint = keras.callbacks.ModelCheckpoint(
                  os.path.join(
                      args.snapshot_path,
                      '{backbone}_{dataset_type}_p{prev_epoch}_{{epoch:02d}}.h5'.format(backbone=args.backbone, dataset_type=args.dataset_type, prev_epoch=args.previous_epoch)
                  ),
                  ## I'm adding these things to always save a model (and overwrite) if it improves the score
                  verbose=1,
                  save_best_only=False,
              )              
        else:
            checkpoint = keras.callbacks.ModelCheckpoint(
                os.path.join(
                    args.snapshot_path,
                    '{backbone}_{dataset_type}_p{prev_epoch}_{{epoch:02d}}_{{loss:.2f}}.h5'.format(backbone=args.backbone, dataset_type=args.dataset_type, prev_epoch=args.previous_epoch)
                ),
                ## I'm adding these things to always save a model (and overwrite) if it improves the score
                verbose=1,
                save_best_only=False,
                monitor="loss",
                mode='min'
            ) 
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor    = 'loss',
        factor     = 0.1,
        patience   = 2,
        verbose    = 1,
        mode       = 'auto',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 0
    ))

    return callbacks


def create_generators(args, preprocess_image):
    """ Create generators for training and validation.
    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    if args.fpn_layers == 5:
      fpn_layers = [3, 4, 5, 6, 7]
    elif args.fpn_layers == 4:
      fpn_layers = [4, 5, 6, 7]
    elif args.fpn_layers == 3:
      fpn_layers = [5, 6, 7]

    common_args = {
        'batch_size'       : args.batch_size,
        'config'           : args.config,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'preprocess_image' : preprocess_image,
        'negative_overlap' : args.neg_overlap,
        'positive_overlap' : args.pos_overlap,
        'train_type'       : args.train_type,
        'fpn_layers'       : fpn_layers,
        'augm'             : args.augm,
    }

    # create random transform generator for augmenting training data
    if args.random_transform:
      transform_generator = random_transform_generator(
          min_rotation=-0.1,
          max_rotation=0.1,
          min_translation=(-0.1, -0.1),
          max_translation=(0.1, 0.1),
          min_shear=-0.1,
          max_shear=0.1,
          min_scaling=(0.9, 0.9),
          max_scaling=(1.1, 1.1),
          flip_x_chance=0.5,
          flip_y_chance=0.5,
      )
    #elif args.random_transform == "augm_a":
    #  transform_generator = random_transform_generator(
    #      min_rotation=-0.5,
    #      max_rotation=0.5,
    #      min_translation=(-0.3, -0.3),
    #      max_translation=(0.3, 0.3),
    #      min_shear=-0.3,
    #      max_shear=0.3,
    #      min_scaling=(0.6, 0.6),
    #      max_scaling=(1.4, 1.4),
    #      flip_x_chance=0.5,
    #      flip_y_chance=0.5,
    #  )
    else:
      transform_generator = random_transform_generator(flip_x_chance=0.5)
      
    if args.dataset_type == 'ead' or args.dataset_type == 'polyp':
        train_generator = CSVGenerator(
            os.path.join(args.data_dir, args.annotations),
            os.path.join(args.data_dir, args.classes),
            transform_generator=transform_generator,
            base_dir=os.path.join(args.data_dir,args.train_dir),
            **common_args
        )

        if args.val_annotations:
            validation_generator = CSVGenerator(
                os.path.join(args.data_dir, args.val_annotations) ,
                os.path.join(args.data_dir, args.classes),
                base_dir=os.path.join(args.data_dir,args.val_dir),
                **common_args
            )
        else:
            validation_generator = None

    else:
        raise ValueError('Invalid data type received: {}'.format(dataset_type))

    return train_generator, validation_generator


def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.
    Args
        parsed_args: parser.parse_args()
    Returns
        parsed_args
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    return parsed_args


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--data-dir', help='Main data directory', dest='data_dir')
    parser.add_argument('--train-dir', help='Train data directory', dest='train_dir', default='train')
    parser.add_argument('--val-dir', help='Val data directory', dest='val_dir', default='val')
    parser.add_argument('--annotations', help='Path to CSV file containing annotations for training.')
    parser.add_argument('--classes', help='Path to a CSV file containing class label mapping.')
    parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')
    parser.add_argument('--snapshot',          help='Resume training from a snapshot.')
    parser.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    parser.add_argument('--weights',           help='Initialize the model with weights from a file.')
    parser.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)
    parser.add_argument('--backbone',         help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',       help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu',        help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force',  help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--steps',            help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--lr',               help='Learning rate.', type=float, default=1e-5)
    parser.add_argument('--snapshot-path',    help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='snapshots')
    parser.add_argument('--tensorboard-dir',  help='Log directory for Tensorboard output', default='logs')
    parser.add_argument('--no-snapshots',     help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',    help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file.')
    parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')
    parser.add_argument('--dataset-type',     help='for validation purposes', default='polyp')
    parser.add_argument('--mode',             help='solely detection or scoring', default='scoring')
    parser.add_argument('--fl-gamma',         help='Gamma value for Focal Loss.', type=float, default=2)
    parser.add_argument('--fl-alpha',         help='Alpha value for Focal Loss.', type=float, default=0.25)
    # parser.add_argument('--aml',  help='Log with AML services', action='store_false')
    parser.add_argument('--previous-epoch',    help='The last epoch that was fully completed on previous training', type=int, default=0)
    parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.4).', default=0.4, type=float)
 
    parser.add_argument('--neg-overlap',  help='Upper IoU Threshold for considering bbox as FP in training.', default=0.4, type=float)
    parser.add_argument('--pos-overlap',  help='Lower IoU Threshold for considering bbox as TP in training..', default=0.5, type=float)
    parser.add_argument('--fpn-layers',   help='Number of FPN Layers to use. Either 4 or 5.', default=5, type=int)
    parser.add_argument('--augm',         help='Which augmentation to use', default=0, type=int)
    #parser.add_argument('--im-count',   help='How many images are in the train set.', default=1800, type=int)
    parser.add_argument('--c-weight',     help='weight of classification loss', default=1, type=int)
    parser.add_argument('--r-weight',     help='weight of regression loss', default=1, type=int)
    parser.add_argument('--p-weight',     help='weight of polyp loss', default=1, type=int)
    parser.add_argument('--loss-weights',     help='weight for different loss functions', default=None, type=str)
    parser.add_argument('--train-type',   help='whether we doing multitask or not', default="single", type=str)
    parser.add_argument('--class-weights',   help='weights of different classes', default=None, )

    # Fit generator arguments
    parser.add_argument('--workers', help='Number of multiprocessing workers. To disable multiprocessing, set workers to 0', type=int, default=1)
    parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit generator.', type=int, default=10)

    return check_args(parser.parse_args(args))


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # create object that stores backbone information
    #backbone = models.backbone(args.backbone)
    backbone = models.backbone(args.backbone)
    

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # create the generators
    #print(args)
    train_generator, validation_generator = create_generators(args, backbone.preprocess_image)
    
    # Log configs
    #run.log('batch-size', args.batch_size)
    #run.log('gamma', args.fl_gamma)
    #run.log('alpha', args.fl_alpha)
    #run.log('lr', args.lr)
    #run.log('neg-overlap', args.neg_overlap)
    #run.log('pos-overlap', args.pos_overlap)
    #run.log('fpn-layers', args.fpn_layers)
    
    if args.class_weights is not None:    
      if args.class_weights == "cw1":
        polyp_weight  = 0.25
        #class_weights = {'classification': {0:a_w, 1:a_w, 2:a_w, 3:a_w, 4:a_w, 5:a_w, 6:a_w, 7:polyp_weight}, 'regression': {0:0.25, 1:0.25, 2:0.25, 3:0.25}}
        #class_weights = {'classification': [polyp_weight, a_w, a_w, a_w, a_w, a_w, a_w, a_w]}
      elif args.class_weights == "cw2":
        polyp_weight  = 0.5
      elif args.class_weights == "cw3":
        polyp_weight  = 0.75
      a_w           = (1-polyp_weight)/7
      #class_weights = {'classification': [polyp_weight, a_w, a_w, a_w, a_w, a_w, a_w, a_w]}  
      class_weights = [polyp_weight, a_w, a_w, a_w, a_w, a_w, a_w, a_w]
    else:
      class_weights = None
    
    if args.loss_weights is None:
      loss_weights = [1, 1]
    elif args.loss_weights == "lw0":
      loss_weights = [1, 1, 1]
    elif args.loss_weights == "lw1":
      loss_weights = [1, 1, 3]
    elif args.loss_weights == "lw2":
      loss_weights = [1, 1, 10]
    elif args.loss_weights == "lw3":
      loss_weights = [1, 1, 20]    
    
    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model            = models.load_model(os.path.join(args.data_dir, args.snapshot), backbone_name=args.backbone)
        training_model   = model
        anchor_params    = None
        if args.config and 'anchor_parameters' in args.config:
            anchor_params = parse_anchor_parameters(args.config)
        prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)
    else:
        if args.weights is None and args.imagenet_weights:
          weights = backbone.download_imagenet()
        else:
          weights = args.weights
        # default to imagenet if nothing else is specified
        ## SO the file that is downloaded is actually only the weights
        ## this means that I should be able to use --weights to give it my own model
        sample_test = np.array([[0.25,0.25,0.25,0.25, 0, 0, 0, 0],[10,10,10,10,10,10,10,10]])
        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            class_weights=class_weights,
            loss_weights=loss_weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone,
            lr=args.lr,
            config=args.config,
            fl_gamma = args.fl_gamma,
            fl_alpha = args.fl_alpha,
            c_weight = args.c_weight,
            r_weight = args.r_weight,
            p_weight = args.p_weight,
            train_type = args.train_type,
            sample_t = sample_test
        )

    # print model summary
    #print(model.summary())

    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'vgg' in args.backbone or 'densenet' in args.backbone:
        train_generator.compute_shapes = make_shapes_callback(model)
        if validation_generator:
            validation_generator.compute_shapes = train_generator.compute_shapes

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        args,
    )

    # Use multiprocessing if workers > 0
    if args.workers > 0:
        use_multiprocessing = True
    else:
        use_multiprocessing = False
        
    temp_df = pd.read_csv(os.path.join(args.data_dir, args.annotations), names=["image_path", "x1", "y1", "x2", "y2", "object_id"])
    im_count = len(set(list(temp_df.image_path)))
   

    # start training
    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=int(im_count/args.batch_size),
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        workers=args.workers,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=args.max_queue_size
        #class_weight=class_weights
    )


if __name__ == '__main__':
   main()