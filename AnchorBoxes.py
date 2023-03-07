from __future__ import division
from bounding_box_utils import convert_coordinates
import logging
import numpy as np
import tensorflow as tf 

logging.basicConfig(level=logging.DEBUG,format='%(levelname)s:%(message)s')

class AnchorBoxes:

    '''
    A Keras layer to create an output tensor containing anchor box coordinates
    and variances based on the input tensor and the passed arguments.
    A set of 2D anchor boxes of different aspect ratios is created for each spatial unit of
    the input tensor. The number of anchor boxes created per unit depends on the arguments
    `aspect_ratios` and `two_boxes_for_ar1`, in the default case it is 4. The boxes
    are parameterized by the coordinate tuple `(xmin, xmax, ymin, ymax)`.
    The logic implemented by this layer is identical to the logic in the module
    `ssd_box_encode_decode_utils.py`.
    The purpose of having this layer in the network is to make the model self-sufficient
    at inference time. Since the model is predicting offsets to the anchor boxes
    (rather than predicting absolute box coordinates directly), one needs to know the anchor
    box coordinates in order to construct the final prediction boxes from the predicted offsets.
    If the model's output tensor did not contain the anchor box coordinates, the necessary
    information to convert the predicted offsets back to absolute coordinates would be missing
    in the model output. The reason why it is necessary to predict offsets to the anchor boxes
    rather than to predict absolute box coordinates directly is explained in `README.md`.
    Input shape:
        4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
        or `(batch, height, width, channels)` if `dim_ordering = 'tf'`.
    Output shape:
        5D tensor of shape `(batch, height, width, n_boxes, 8)`. The last axis contains
        the four anchor box coordinates and the four variance values for each box.
    '''

    def __init__(self,
                 img_height,
                 img_width,
                 this_scale,
                 next_scale,
                 aspect_ratios=[0.5,1.0,2.0],
                 two_boxes_for_ar1=True,
                 this_steps=None,
                 this_offsets=None,
                 clip_boxes=False,
                 variances=[0.1,1,0.2,0.2],
                 coords='centroids',
                 normalize_coords=False,
                 **kwargs):

        '''
        All arguments need to be set to the same values as in the box encoding process, otherwise the behavior is undefined.
        Some of these arguments are explained in more detail in the documentation of the `SSDBoxEncoder` class.
        Arguments:
            img_height (int): The height of the input images.
            img_width (int): The width of the input images.
            this_scale (float): A float in [0, 1], the scaling factor for the size of the generated anchor boxes
                as a fraction of the shorter side of the input image.
            next_scale (float): A float in [0, 1], the next larger scaling factor. Only relevant if
                `self.two_boxes_for_ar1 == True`.
            aspect_ratios (list, optional): The list of aspect ratios for which default boxes are to be
                generated for this layer.
            two_boxes_for_ar1 (bool, optional): Only relevant if `aspect_ratios` contains 1.
                If `True`, two default boxes will be generated for aspect ratio 1. The first will be generated
                using the scaling factor for the respective layer, the second one will be generated using
                geometric mean of said scaling factor and next bigger scaling factor.
            clip_boxes (bool, optional): If `True`, clips the anchor box coordinates to stay within image boundaries.
            variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
                its respective variance value.
            coords (str, optional): The box coordinate format to be used internally in the model (i.e. this is not the input format
                of the ground truth labels). Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width, and height),
                'corners' for the format `(xmin, ymin, xmax,  ymax)`, or 'minmax' for the format `(xmin, xmax, ymin, ymax)`.
            normalize_coords (bool, optional): Set to `True` if the model uses relative instead of absolute coordinates,
                i.e. if the model predicts box coordinates within [0,1] instead of absolute coordinates.
        '''

        if (this_scale < 0) or (next_scale < 0) or (this_scale > 1):
            raise ValueError("this_scale` must be in [0, 1] and next_scale must be >0, but this_scale == {}, next_scale == {}".format(this_scale, next_scale))

        if len(variances) != 4:
            raise ValueError("4 variance values must be pased, but {} values were received".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        self.img_height = img_height
        self.img_width = img_width
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.this_steps = this_steps
        self.this_offsets = this_offsets
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.coords = coords
        self.normalize_coords = normalize_coords

        # Compute the number of boxes per cell
        if (1 in aspect_ratios) and two_boxes_for_ar1:
            self.n_boxes = len(aspect_ratios) + 1
        else:
            self.n_boxes=len(aspect_ratios)

        super(AnchorBoxes, self).__init__(**kwargs)

    def call(self,x,mask=None):
        '''
        Return an anchor box tensor based on the shape of the input tensor.
        The logic implemented here is identical to the logic in the module `ssd_box_encode_decode_utils.py`.
        Note that this tensor does not participate in any graph computations at runtime. It is being created
        as a constant once during graph creation and is just being output along with the rest of the model output
        during runtime. Because of this, all logic is implemented as Numpy array operations and it is sufficient
        to convert the resulting Numpy array into a Keras tensor at the very end before outputting it.
        Arguments:
            x (tensor): 4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
                or `(batch, height, width, channels)` if `dim_ordering = 'tf'`. The input for this
                layer must be the output of the localization predictor layer.
        '''

        # Compute box width and height for each aspect ratio
        # The shorter side of the image will be used to compute w and h using `scale` and aspect_ratios.
        size = min(self.img_height,self.img_width)
        # Compute the box widths and heights for all aspect ratios
        wh_list = []
        for ar in self.aspect_ratios:
            if (ar == 1):
                # Compute the regular anchor box for aspect ratio 1
                box_height = box_width = self.this_scale * size
                wh_list.append((box_width,box_height))
                if self.two_boxes_for_ar1:
                    # Compute one slightly larger version using the geometric mean f this scale value and the next
                    box_height = box_width = np.sqrt(self.this_scale * self.next_scale) * size
                    wh_list.append((box_height,box_width))
            else:
                box_height = self.this_scale * size / np.sqrt(ar)
                box_width = self.this_scale * size * np.sqrt(ar)
                wh_list.append((box_width,box_height))
        wh_list = np.array(wh_list)

        # We need the shape of the input tensor
        batch_size, feature_map_height, feature_map_width, feature_map_channels=x.get_shape().as_list()

        # Compute the grid of box center points. They are identical for all aspect ratios 


        # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
        if (self.this_steps is None):
            step_height = self.img_height / feature_map_height
            step_width = self.img_width / feature_map_width
        else:
            if isinstance(self.this_steps,(list,tuple)) and (len(self.this_steps) == 2):
                step_height = self.this_steps[0]
                step_width = self.this_steps[1]
            elif isinstance(self.this_steps,(int,float)):
                step_height = self.this_steps
                step_width = self.this_steps
        # Compute the offsets, i.e. at what pixel values the first anchor box center point will be from the top and from the left of the image.
        if (self.this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(self.this_offsets, (list, tuple)) and (len(self.this_offsets) == 2):
                offset_height = self.this_offsets[0]
                offset_width = self.this_offsets[1]
            elif isinstance(self.this_offsets, (int, float)):
                offset_height = self.this_offsets
                offset_width = self.this_offsets
        # Now that we have the offsets and step sizes, compute the grid of anchor box center points

        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_height - 1) * step_height, feature_map_height)
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_width - 1) * step_width, feature_map_width)
        cx_grid = np.expand_dims(cx, -1) # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy, -1) # This is necessary for np.tile() to do what we want further down

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))


        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes)) # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes)) # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Set h

        # Convert (cx, cy, w, h) to (xmin,ymin,xmax,ymax)
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        # If clip_boxes is enabled, clip the coordinates to lie within the image boundaries

        if self.clip_boxes:
            x_coords = boxes_tensor[:,:,:,[0,2]]
            x_coords[x_coords >= self.img_width] = self.img_width -1
            x_coords[x_coords < 0] = 0
            boxes_tensors[:,:,:,[0,2]] = x_coords
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:,:,:,[1, 3]] = y_coords

        # If normalize_coords is enabled , normalize the coordinates to be within [0,1]
        if self.normalize_coords:
            boxes_tensor[:,:,:,[0,2]] /= self.img_width
            boxes_tensor[:,:,:,[1,3]] /= self.img_height

        # TODO: Implement box limiting directly for `(cx, cy, w, h)` so that we don't have to unnecessarily convert back and forth.
        if self.coords == 'centroids':
            # Convert `(xmin, ymin, xmax, ymax)` back to `(cx, cy, w, h)`.
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids', border_pixels='half')
        elif self.coords == 'minmax':
            # Convert `(xmin, ymin, xmax, ymax)` to `(xmin, xmax, ymin, ymax).
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2minmax', border_pixels='half')

        # Create a tensor to contain the variances and append it to `boxes_tensor`. This tensor has the same shape
        # as `boxes_tensor` and simply contains the same 4 variance values for every position in the last axis.
        variances_tensor = np.zeros_like(boxes_tensor) # Has shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        variances_tensor += self.variances # Long live broadcasting
        # Now `boxes_tensor` becomes a tensor of shape `(feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

        # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
        # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)

        boxes_tensor=tf.constant(value=boxes_tensor,dtype=tf.float32)

        return boxes_tensor


#Testing
if __name__=='__main__':

    img_size = (300,300,3)
    img_height = img_size[1]
    img_width = img_size[0]
    n_classes = 20
    mode ='training'
    l2_regularization = 0.0005
    min_scale = None
    max_scale = None
    scales = [0.1,0.2,0.37,0.54,0.71,0.88,1.05]
    aspect_ratios_global = None
    aspect_ratios = [[1.0,2.0,0.5],
                         [1.0,2.0,0.5,3.0,1.0/3.0],
                         [1.0,2.0,0.5,3.0,1.0/3.0],
                         [1.0,2.0,0.5,3.0,1.0/3.0],
                         [1.0,2.0,0.5],
                         [1.0,2.0,0.5]]
    two_boxes_for_ar1 = True
    steps = [8,16,32,64,100,300]
    offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    clip_boxes = False
    variances = [0.1,0.1,0.2,0.2]
    coords = 'centroids'
    normalize_coords = True
    subtract_mean = [123,117,104]
    divide_by_stddev = None
    confidence_thresh = 0.01
    iou_thrshold = 0.45
    top_k = 200
    nms_max_output_size = 400
    return_predictor_sizes = False

    # Create NumPy Location layers , in order to AnchorBoxes can get the Layer widths and heights

    conv4_3_norm_mbox_loc = np.reshape(np.random.rand(1,1*38*38*16),(1,38,38,16))

    fc7_mbox_loc = np.reshape(np.random.rand(1,1*19*19*16),(1,19,19,16))

    conv6_2_mbox_loc = np.reshape(np.random.rand(1,1*10*10*16),(1,10,10,16))

    conv7_2_mbox_loc = np.reshape(np.random.rand(1,1*5*5*24),(1,5,5,24))

    conv8_2_mbox_loc = np.reshape(np.random.rand(1,1*3*3*24),(1,3,3,24))

    conv9_2_mbox_loc = np.reshape(np.random.rand(1,1*1*1*24),(1,1,1,24))


    # Conver NumPy Location Layers to TensorFlow Layers

    conv4_3_norm_mbox_loc = tf.Variable(conv4_3_norm_mbox_loc,dtype=tf.float32)

    fc7_mbox_loc = tf.Variable(fc7_mbox_loc,dtype=tf.float32)

    conv6_2_mbox_loc = tf.Variable(conv6_2_mbox_loc,dtype=tf.float32)

    conv7_2_mbox_loc = tf.Variable(conv7_2_mbox_loc,dtype=tf.float32)

    conv8_2_mbox_loc = tf.Variable(conv8_2_mbox_loc,dtype=tf.float32)

    conv9_2_mbox_loc = tf.Variable(conv9_2_mbox_loc,dtype=tf.float32)




    ### Generate the anchor boxes

    # Output shape of anchors: (batch, height, width, n_boxes, 8)
    conv4_3_norm_mbox_priorbox = AnchorBoxes(img_height=img_height,img_width=img_width,this_scale=scales[0],next_scale=scales[1],aspect_ratios=aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1,this_steps=steps[0],this_offsets=offsets[0],clip_boxes=clip_boxes,
                                             variances=variances,coords=coords,normalize_coords=normalize_coords).call(conv4_3_norm_mbox_loc)

    logging.debug('conv4_3_norm_mbox_priorbox has a shape {}'.format(conv4_3_norm_mbox_priorbox.get_shape()))

    fc7_mbox_priorbox = AnchorBoxes(img_height=img_height,img_width=img_width,this_scale=scales[1],next_scale=scales[2],aspect_ratios=aspect_ratios[1],
                                             two_boxes_for_ar1=two_boxes_for_ar1,this_steps=steps[1],this_offsets=offsets[1],clip_boxes=clip_boxes,
                                             variances=variances,coords=coords,normalize_coords=normalize_coords).call(fc7_mbox_loc)

    logging.debug('fc7_mbox_priorbox has a shape {}'.format(fc7_mbox_priorbox.get_shape()))

    conv6_2_norm_mbox_priorbox = AnchorBoxes(img_height=img_height,img_width=img_width,this_scale=scales[2],next_scale=scales[3],aspect_ratios=aspect_ratios[2],
                                         two_boxes_for_ar1=two_boxes_for_ar1,this_steps=steps[2],this_offsets=offsets[2],clip_boxes=clip_boxes,
                                         variances=variances,coords=coords,normalize_coords=normalize_coords).call(conv6_2_mbox_loc)

    logging.debug('conv6_2_norm_mbox_priorbox has a shape {}'.format(conv6_2_norm_mbox_priorbox.get_shape()))

    conv7_2_norm_mbox_priorbox = AnchorBoxes(img_height=img_height,img_width=img_width,this_scale=scales[3],next_scale=scales[4],aspect_ratios=aspect_ratios[3],
                                         two_boxes_for_ar1=two_boxes_for_ar1,this_steps=steps[3],this_offsets=offsets[3],clip_boxes=clip_boxes,
                                         variances=variances,coords=coords,normalize_coords=normalize_coords).call(conv7_2_mbox_loc)

    logging.debug('conv7_2_norm_mbox_priorbox has a shape {}'.format(conv7_2_norm_mbox_priorbox.get_shape()))

    conv8_2_norm_mbox_priorbox = AnchorBoxes(img_height=img_height,img_width=img_width,this_scale=scales[4],next_scale=scales[5],aspect_ratios=aspect_ratios[4],
                                     two_boxes_for_ar1=two_boxes_for_ar1,this_steps=steps[4],this_offsets=offsets[4],clip_boxes=clip_boxes,
                                     variances=variances,coords=coords,normalize_coords=normalize_coords).call(conv8_2_mbox_loc)

    logging.debug('conv8_2_norm_mbox_priorbox has a shape {}'.format(conv8_2_norm_mbox_priorbox.get_shape()))

    conv9_2_norm_mbox_priorbox = AnchorBoxes(img_height=img_height,img_width=img_width,this_scale=scales[5],next_scale=scales[6],aspect_ratios=aspect_ratios[5],
                                     two_boxes_for_ar1=two_boxes_for_ar1,this_steps=steps[5],this_offsets=offsets[5],clip_boxes=clip_boxes,
                                     variances=variances,coords=coords,normalize_coords=normalize_coords).call(conv9_2_mbox_loc)

    logging.debug('conv9_2_norm_mbox_priorbox has a shape {}'.format(conv9_2_norm_mbox_priorbox.get_shape()))






