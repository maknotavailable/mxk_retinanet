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
import numpy as np
import backend
from itertools import product


def focal(c_weight=1, alpha=0.25, gamma=2.0, weights_list=None):

    """ Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """
    def _focal(y_true, y_pred):
        """ Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        labels         = y_true[:, :, :-1]
        anchor_state   = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred

        # filter out "ignore" anchors
        indices        = backend.where(keras.backend.not_equal(anchor_state, -1))
        labels         = backend.gather_nd(labels, indices)
        classification = backend.gather_nd(classification, indices)
        
        if weights_list is not None:

            # adding my own weights
            if keras.backend.image_data_format() == 'channels_first':
            axis = 1
            else:
            axis = -1
            
            classSelectors = keras.backend.argmax(labels, axis=axis) 
            classSelectors = [keras.backend.equal(np.int64(i), classSelectors) for i in range(len(weights_list))]
            classSelectors = [keras.backend.cast(x, keras.backend.floatx()) for x in classSelectors]
            weights = [sel * w for sel,w in zip(classSelectors, weights_list)] 
            weightMultiplier = weights[0]
            for i in range(1, len(weights)):
                weightMultiplier = weightMultiplier + weights[i]
            weightMultiplier = keras.backend.expand_dims(weightMultiplier, 1)
            weightMultiplier = keras.backend.tile(weightMultiplier, [1,8])
            
            
        """
        weights_array = np.ones((8,8))
        i = 0
        for w in weights_list:
          weights_array[i,:] = w
          i += 1
        
        nb_cl = len(weights_array)
        final_mask = keras.backend.zeros_like(classification[:, 0])
        y_pred_max = keras.backend.max(classification, axis=1)
        y_pred_max = keras.backend.reshape(y_pred_max, (keras.backend.shape(classification)[0], 1))
        #y_pred_max = keras.backend.expand_dims(y_pred_max, 1)
        y_pred_max_mat = keras.backend.equal(classification, y_pred_max)
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
          final_mask += (keras.backend.cast(weights_array[c_t, c_p],tf.float32) * keras.backend.cast(y_pred_max_mat[:, c_p] ,tf.float32)* keras.backend.cast(labels[:, c_t],tf.float32))
        """
        
            
        # add my own weights
        #if weight_list:
          #weight_factor = keras.backend.zeros_like(labels)
        #  weight_factor = keras.backend.ones_like(labels)
        #  for i in range(len(weight_list)):
        #    weight_factor = backend.scatter_update(weight_factor[:,:])
        #    weight_np[:,:,i] = weight_list[i]
        #  weight_factor    = keras.backend.variable(weight_np)
        #else:
        #  weight_factor = keras.backend.ones_like(labels)


        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = backend.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = backend.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        #cls_loss = weight_factor * focal_weight * keras.backend.binary_crossentropy(labels, classification)
        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification) * weightMultiplier
        
        # compute the normalizer: the number of positive anchors
        normalizer = backend.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        return c_weight*(keras.backend.sum(cls_loss) / normalizer)

    return _focal
  
def focal2(p_weight=1, alpha=0.25, gamma=2.0):

    """ Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """
    def _focal2(y_true, y_pred):
        """ Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        labels         = y_true[:, :, :-1]
        anchor_state   = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred

        # filter out "ignore" anchors
        indices        = backend.where(keras.backend.not_equal(anchor_state, -1))
        labels         = backend.gather_nd(labels, indices)
        classification = backend.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = backend.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = backend.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = backend.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        return p_weight*(keras.backend.sum(cls_loss) / normalizer)

    return _focal2


def smooth_l1(r_weight=1, sigma=3.0):
    """ Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices           = backend.where(keras.backend.equal(anchor_state, 1))
        regression        = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return r_weight*(keras.backend.sum(regression_loss) / normalizer)

    return _smooth_l1
