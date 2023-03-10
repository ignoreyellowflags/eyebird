from __future__ import division
import numpy as np
import tensorflow as tf 
from AnchorBoxes import AnchorBoxes
# from keras_layers.keras_layer_L2Normalization import L2Normalization
# from keras_layers.keras_layer_DecodeDetections import DecodeDetections
# from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
def ssd_300(image_size,
            n_classes,
            mode ='training',
            l2_regularization = 0.0005,
            min_scal = None,
            max_scale = None,
            scales = None,
            aspect_ratios_global = None,
            aspect_ratios_per_layer=[[1.0,2.0,0.5],
                                     [1.0,2.0,0.5,3.0,1.0/3.0],
                                     [1.0,2.0,0.5,3.0,1.0/3.0],
                                     [1.0,2.0,0.5,3.0,1.0/3.0],
                                     [1.0,2.0,0.5],
                                     [1.0,2.0,0.5],
                                     [1.0,2.0,0.5]],
            two_boxes_for_ar1 = True,
            steps = [8,16,32,64,100,300],
            offsets = None,
            clip_boxes = False,
            variances = [0.1,0.1,0.2,0.2],
            coords = 'centroids',
            normalize_coords = True,
            subtract_mean = [123,117,104],
            divide_by_stddev = None,
            confidence_thresh = 0.01,
            iou_thrshold = 0.45,
            top_k = 200,
            nms_max_output_size = 400,
            return_predictor_sizes = False):

    '''
    Build a Keras model with SSD300 architecture, see references.
    The base network is a reduced atrous VGG-16, extended by the SSD architecture,
    as described in the paper.
    Most of the arguments that this function takes are only needed for the anchor
    box layers. In case you're training the network, the parameters passed here must
    be the same as the ones used to set up `SSDBoxEncoder`. In case you're loading
    trained weights, the parameters passed here must be the same as the ones used
    to produce the trained weights.
    Some of these arguments are explained in more detail in the documentation of the
    `SSDBoxEncoder` class.
    Note: Requires Keras v2.0 or later. Currently works only with the
    TensorFlow backend (v1.0 or later).
    Arguments:
        image_size (tuple): The input image size in the format `(height, width, channels)`.
        n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
        mode (str, optional): One of 'training', 'inference' and 'inference_fast'. In 'training' mode,
            the model outputs the raw prediction tensor, while in 'inference' and 'inference_fast' modes,
            the raw predictions are decoded into absolute coordinates and filtered via confidence thresholding,
            non-maximum suppression, and top-k filtering. The difference between latter two modes is that
            'inference' follows the exact procedure of the original Caffe implementation, while
            'inference_fast' uses a faster prediction decoding procedure.
        l2_regularization (float, optional): The L2-regularization rate. Applies to all convolutional layers.
            Set to zero to deactivate L2-regularization.
        min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images.
        max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images. All scaling factors between the smallest and the
            largest will be linearly interpolated. Note that the second to last of the linearly interpolated
            scaling factors will actually be the scaling factor for the last predictor layer, while the last
            scaling factor is used for the second box for aspect ratio 1 in the last predictor layer
            if `two_boxes_for_ar1` is `True`.
        scales (list, optional): A list of floats containing scaling factors per convolutional predictor layer.
            This list must be one element longer than the number of predictor layers. The first `k` elements are the
            scaling factors for the `k` predictor layers, while the last element is used for the second box
            for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`. This additional
            last scaling factor must be passed either way, even if it is not being used. If a list is passed,
            this argument overrides `min_scale` and `max_scale`. All scaling factors must be greater than zero.
        aspect_ratios_global (list, optional): The list of aspect ratios for which anchor boxes are to be
            generated. This list is valid for all prediction layers.
        aspect_ratios_per_layer (list, optional): A list containing one aspect ratio list for each prediction layer.
            This allows you to set the aspect ratios for each predictor layer individually, which is the case for the
            original SSD300 implementation. If a list is passed, it overrides `aspect_ratios_global`.
        two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratio lists that contain 1. Will be ignored otherwise.
            If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
            using the scaling factor for the respective layer, the second one will be generated using
            geometric mean of said scaling factor and next bigger scaling factor.
        steps (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either ints/floats or tuples of two ints/floats. These numbers represent for each predictor layer how many
            pixels apart the anchor box center points should be vertically and horizontally along the spatial grid over
            the image. If the list contains ints/floats, then that value will be used for both spatial dimensions.
            If the list contains tuples of two ints/floats, then they represent `(step_height, step_width)`.
            If no steps are provided, then they will be computed such that the anchor box center points will form an
            equidistant grid within the image dimensions.
        offsets (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either floats or tuples of two floats. These numbers represent for each predictor layer how many
            pixels from the top and left boarders of the image the top-most and left-most anchor box center points should be
            as a fraction of `steps`. The last bit is important: The offsets are not absolute pixel values, but fractions
            of the step size specified in the `steps` argument. If the list contains floats, then that value will
            be used for both spatial dimensions. If the list contains tuples of two floats, then they represent
            `(vertical_offset, horizontal_offset)`. If no offsets are provided, then they will default to 0.5 of the step size.
        clip_boxes (bool, optional): If `True`, clips the anchor box coordinates to stay within image boundaries.
        variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
            its respective variance value.
        coords (str, optional): The box coordinate format to be used internally by the model (i.e. this is not the input format
            of the ground truth labels). Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width,
            and height), 'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
        normalize_coords (bool, optional): Set to `True` if the model is supposed to use relative instead of absolute coordinates,
            i.e. if the model predicts box coordinates within [0,1] instead of absolute coordinates.
        subtract_mean (array-like, optional): `None` or an array-like object of integers or floating point values
            of any shape that is broadcast-compatible with the image shape. The elements of this array will be
            subtracted from the image pixel intensity values. For example, pass a list of three integers
            to perform per-channel mean normalization for color images.
        divide_by_stddev (array-like, optional): `None` or an array-like object of non-zero integers or
            floating point values of any shape that is broadcast-compatible with the image shape. The image pixel
            intensity values will be divided by the elements of this array. For example, pass a list
            of three integers to perform per-channel standard deviation normalization for color images.
        swap_channels (list, optional): Either `False` or a list of integers representing the desired order in which the input
            image channels should be swapped.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage.
        iou_threshold (float, optional): A float in [0,1]. All boxes that have a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box's confidence score.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage.
        nms_max_output_size (int, optional): The maximal number of predictions that will be left over after the NMS stage.
        return_predictor_sizes (bool, optional): If `True`, this function not only returns the model, but also
            a list containing the spatial dimensions of the predictor layers. This isn't strictly necessary since
            you can always get their sizes easily via the Keras API, but it's convenient and less error-prone
            to get them this way. They are only relevant for training anyway (SSDBoxEncoder needs to know the
            spatial dimensions of the predictor layers), for inference you don't need them.
    Returns:
        model: The Keras SSD300 model.
        predictor_sizes (optional): A Numpy array containing the `(height, width)` portion
            of the output tensor shape for each convolutional predictor layer. During
            training, the generator function needs this in order to transform
            the ground truth labels into tensors of identical structure as the
            output tensors of the model, which is in turn needed for the cost
            function.
    References:
        https://arxiv.org/abs/1512.02325v5
    '''

    n_predictor_layers = 6 #The number of predictor conv layers in the network is 6 for the original SSD300
    n_classes += 1 #Account for the background class
    l2_reg = l2_regularization #Make the internal name shorter
    img_height,img_width,img_channels = img_size[0],img_size[1],img_size[2]

    ###############################

    # Get a few exceptions out of the way

    ###############################

	if aspect_ratios_global is None and aspect_ratios_per_layer is None:
	    raise ValueError('aspect_ratios_global and aspect_reatios_per_layer cannot both be None.At least one needs to be specified')
	if aspect_ratios_per_layer:
	    if len(aspect_ratios_per_layer) != n_predictor_layers:
	        raise ValueError('It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}'.format(n_predictor_layers,len(aspect_ratios_per_layer)))

	if (min_scale is None or max_scale is None) and scales is None:
	    raise ValueError('Either min_scale and max_scale or scales need to be specified')
	if scales:
	    if len(scales) != n_predictor_layers + 1:
	        raise ValueError('''It must be either scales is None or len(scales) == {},but len(scales) == {}'''.format(n_predictor_layers+1,len(scales)))
	else: # IF no explicit list of scaling factors was passed,compute the list of scaling factors from min_scale and max_scale
	    scales = np.linspace(min_scale,max_scale,n_predictor_layers + 1)

	if len(variances) != 4:
	    raise ValueError('4 variances values must be passed,but {} values were recieved'.format(len(variances)))
	variances = np.array(variances)
	if np.any(variances <= 0):
	    raise ValueError('All varinces must be > 0,but the variances given are {}'.format(variances))


	if (not (steps is None)) and (len(steps) != n_predictor_layers):
	    raise ValueError('You must provide at least one step value per predictor layer')

	if (not (offsets is None)) and (len(offsets) != n_predictors_layers):
	    raise ValueError('You must provide at least one offset value per predictor layer')

	 ###############################

	# Compute the anchor box parameters

	###############################

	# Set the aspect ratios for each predictor layer
	if aspect_ratios_per_layer:
	    aspect_ratios=aspect_ratios_per_layer
	else:
	    aspect_ratios = [aspect_ratios_global] * n_predictor_layers

	# Compute the number of boxes to be predicted per cell for each predictor layer
	# We need this so that we know how many channels the predictor layers need to have

	if aspect_ratios_per_layer:
	    n_boxes=[]
	    for  ar in aspect_ratios_per_layer:
	        if (1 in ar) & two_boxes_for_ar1:
	            n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
	        else:
	            n_boxes.append(len(ar))
	else:# If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
	    if (1 in aspect_ratios_global) & two_boxes_for_ar1:
	        n_boxes = len(aspect_ratios_global) + 1
	    else:
	        n_boxes = len(aspect_ratios_global)
	    n_boxes = [n_boxes] * n_predictor_layers

	if steps is None:
	    steps = [None] * n_predictor_layers
	if offsets is None:
	    offsets = [None] * n_predictor_layers




	input_layer = tf.reshape(tf.Variable(img,tf.float32),[-1,300,300,3])


	conv1_1=tf.layers.conv2d(inputs=input_layer,filters=64,kernel_size=[3,3],padding='same',activation=tf.nn.relu) #(1, 300, 300, 64)

	conv1_2=tf.layers.conv2d(inputs=conv1_1,filters=64,kernel_size=[3,3],padding='same',activation=tf.nn.relu) #(1, 300, 300, 64)

	pool1=tf.layers.max_pooling2d(inputs=conv1_2,pool_size=[2,2],strides=2,padding='same') #(1, 150, 150, 64)


	conv2_1=tf.layers.conv2d(inputs=pool1,filters=128,kernel_size=[3,3],padding='same',activation=tf.nn.relu) #(1, 150, 150, 128)

	conv2_2=tf.layers.conv2d(inputs=conv2_1,filters=128,kernel_size=[3,3],padding='same',activation=tf.nn.relu) #(1, 150, 150, 128)

	pool2=tf.layers.max_pooling2d(inputs=conv2_2,pool_size=[2,2],strides=2,padding='same') #(1, 75, 75, 128)


	conv3_1=tf.layers.conv2d(inputs=pool2,filters=256,kernel_size=[3,3],padding='same',activation=tf.nn.relu) #(1, 75, 75, 256)

	conv3_2=tf.layers.conv2d(inputs=conv3_1,filters=256,kernel_size=[3,3],padding='same',activation=tf.nn.relu) #(1, 75, 75, 256)

	conv3_3=tf.layers.conv2d(inputs=conv3_2,filters=256,kernel_size=[3,3],padding='same',activation=tf.nn.relu) #(1, 75, 75, 256)

	pool3=tf.layers.max_pooling2d(inputs=conv3_3,pool_size=[2,2],strides=2,padding='same') #(1, 38, 38, 256)


	conv4_1=tf.layers.conv2d(inputs=pool3,filters=512,kernel_size=[3,3],padding='same',activation=tf.nn.relu) # (1, 38, 38, 512)

	conv4_2=tf.layers.conv2d(inputs=conv4_1,filters=512,kernel_size=[3,3],padding='same',activation=tf.nn.relu) # (1, 38, 38, 512)

	conv4_3=tf.layers.conv2d(inputs=conv4_2,filters=512,kernel_size=[3,3],padding='same',activation=tf.nn.relu) # (1, 38, 38, 512)

	pool4=tf.layers.max_pooling2d(inputs=conv4_3,pool_size=[2,2],strides=2,padding='same') # (1, 19, 19, 512)


	conv5_1=tf.layers.conv2d(inputs=pool4,filters=512,kernel_size=[3,3],padding='same',activation=tf.nn.relu) # (1, 19, 19, 512)

	conv5_2=tf.layers.conv2d(inputs=conv5_1,filters=512,kernel_size=[3,3],padding='same',activation=tf.nn.relu) # (1, 19, 19, 512)

	conv5_3=tf.layers.conv2d(inputs=conv5_2,filters=512,kernel_size=[3,3],padding='same',activation=tf.nn.relu) # (1, 19, 19, 512)

	pool5=tf.layers.max_pooling2d(inputs=conv5_3,pool_size=[3,3],strides=1,padding='same') # (1, 19, 19, 512)



	fc6=tf.layers.conv2d(inputs=pool5,filters=1024,kernel_size=[3,3],padding='same',activation=tf.nn.relu) # (1, 19, 19, 1024)

	fc7=tf.layers.conv2d(inputs=fc6,filters=1024,kernel_size=[1,1],padding='same',activation=tf.nn.relu) # (1, 19, 19, 1024)

	conv6_1=tf.layers.conv2d(inputs=fc7,filters=256,kernel_size=[1,1],padding='same',activation=tf.nn.relu) # (1, 19, 19, 256)

	conv6_1=tf.pad(conv6_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT") # (1, 19, 19, 256)

	conv6_2=tf.layers.conv2d(inputs=conv6_1,filters=512,kernel_size=[3,3],strides=(2,2),padding='valid',activation=tf.nn.relu) # (1, 10, 10, 512)

	conv7_1=tf.layers.conv2d(inputs=conv6_2,filters=128,kernel_size=[1,1],padding='same',activation=tf.nn.relu)  # (1, 10, 10, 128)

	conv7_1=tf.pad(conv7_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT") # (1, 12, 12, 128)

	conv7_2=tf.layers.conv2d(inputs=conv7_1,filters=256,kernel_size=[3,3],strides=(2,2),padding='valid',activation=tf.nn.relu) # (1, 5, 5, 256)


	conv8_1=tf.layers.conv2d(inputs=conv7_2,filters=128,kernel_size=[1,1],padding='same',activation=tf.nn.relu) # (1, 5, 5, 128)

	conv8_2=tf.layers.conv2d(inputs=conv8_1,filters=256,kernel_size=[3,3],padding='valid',strides=1,activation=tf.nn.relu) # (1, 3, 3, 256)

	conv9_1=tf.layers.conv2d(inputs=conv8_2,filters=128,kernel_size=[1,1],padding='same',activation=tf.nn.relu) # (1, 3, 3, 128)

	conv9_2=tf.layers.conv2d(inputs=conv9_1,filters=256,kernel_size=[3,3],padding='valid',strides=1,activation=tf.nn.relu) # (1, 1, 1, 256)

	# Feed conv4_3 into the L2 normalization layer
	conv4_3_norm=tf.nn.l2_normalize(conv4_3,dim=0)

	# We precidt n_classes confidence values for each box, hence the confidence predictors have depth n_boxes * n_classes
	# Output shape of the confidence layers: (batch, height, width, n_boxes * n_classes)

	conv4_3_norm_mbox_conf=tf.layers.conv2d(inputs=conv4_3_norm,filters=n_boxes[0]*n_classes,kernel_size=[3,3],padding='same',activation=tf.nn.relu)

	fc7_mbox_conf=tf.layers.conv2d(inputs=fc7,filters=n_boxes[1]*n_classes,kernel_size=[3,3],padding='same',activation=tf.nn.relu)

	conv6_2_mbox_conf=tf.layers.conv2d(inputs=conv6_2,filters=n_boxes[2]*n_classes,kernel_size=[3,3],padding='same',activation=tf.nn.relu)

	conv7_2_mbox_conf=tf.layers.conv2d(inputs=conv7_2,filters=n_boxes[3]*n_classes,kernel_size=[3,3],padding='same',activation=tf.nn.relu)

	conv8_2_mbox_conf=tf.layers.conv2d(inputs=conv8_2,filters=n_boxes[4]*n_classes,kernel_size=[3,3],padding='same',activation=tf.nn.relu)

	conv9_2_mbox_conf=tf.layers.conv2d(inputs=conv9_2,filters=n_boxes[5]*n_classes,kernel_size=[3,3],padding='same',activation=tf.nn.relu)

	# We predict 4 box coordinates for each box, hence the localization predictors have depth n_boxes * 4
	# Output shape of the localization layers: (batch, height, width, n_boxes * 4)

	#LOCATION LEVEL

	conv4_3_norm_mbox_loc=tf.layers.conv2d(inputs=conv4_3_norm,filters=n_boxes[0]*4,kernel_size=[3,3],padding='same',activation=tf.nn.relu)

	fc7_mbox_loc=tf.layers.conv2d(inputs=fc7,filters=n_boxes[1]*4,kernel_size=[3,3],padding='same',activation=tf.nn.relu)

	conv6_2_mbox_loc=tf.layers.conv2d(inputs=conv6_2,filters=n_boxes[2]*4,kernel_size=[3,3],padding='same',activation=tf.nn.relu)

	conv7_2_mbox_loc=tf.layers.conv2d(inputs=conv7_2,filters=n_boxes[3]*4,kernel_size=[3,3],padding='same',activation=tf.nn.relu)

	conv8_2_mbox_loc=tf.layers.conv2d(inputs=conv8_2,filters=n_boxes[4]*4,kernel_size=[3,3],padding='same',activation=tf.nn.relu)

	conv9_2_mbox_loc=tf.layers.conv2d(inputs=conv9_2,filters=n_boxes[5]*4,kernel_size=[3,3],padding='same',activation=tf.nn.relu)


	### Generate the anchor boxes

	# Output shape of anchors: (batch, height, width, n_boxes, 8)
	conv4_3_norm_mbox_priorbox = AnchorBoxes(img_height=img_height,img_width=img_width,this_scale=scales[0],next_scale=scales[1],aspect_ratios=aspect_ratios[0],
	                                         two_boxes_for_ar1=two_boxes_for_ar1,this_steps=steps[0],this_offsets=offsets[0],clip_boxes=clip_boxes,
	                                         variances=variances,coords=coords,normalize_coords=normalize_coords).call(conv4_3_norm_mbox_loc)

	fc7_mbox_priorbox = AnchorBoxes(img_height=img_height,img_width=img_width,this_scale=scales[1],next_scale=scales[2],aspect_ratios=aspect_ratios[1],
	                                         two_boxes_for_ar1=two_boxes_for_ar1,this_steps=steps[1],this_offsets=offsets[1],clip_boxes=clip_boxes,
	                                         variances=variances,coords=coords,normalize_coords=normalize_coords).call(fc7_mbox_loc)

	conv6_2_norm_mbox_priorbox = AnchorBoxes(img_height=img_height,img_width=img_width,this_scale=scales[2],next_scale=scales[3],aspect_ratios=aspect_ratios[2],
	                                     two_boxes_for_ar1=two_boxes_for_ar1,this_steps=steps[2],this_offsets=offsets[2],clip_boxes=clip_boxes,
	                                     variances=variances,coords=coords,normalize_coords=normalize_coords).call(conv6_2_mbox_loc)

	conv7_2_norm_mbox_priorbox = AnchorBoxes(img_height=img_height,img_width=img_width,this_scale=scales[3],next_scale=scales[4],aspect_ratios=aspect_ratios[3],
	                                     two_boxes_for_ar1=two_boxes_for_ar1,this_steps=steps[3],this_offsets=offsets[3],clip_boxes=clip_boxes,
	                                     variances=variances,coords=coords,normalize_coords=normalize_coords).call(conv7_2_mbox_loc)

	conv8_2_norm_mbox_priorbox = AnchorBoxes(img_height=img_height,img_width=img_width,this_scale=scales[4],next_scale=scales[5],aspect_ratios=aspect_ratios[4],
	                                 two_boxes_for_ar1=two_boxes_for_ar1,this_steps=steps[4],this_offsets=offsets[4],clip_boxes=clip_boxes,
	                                 variances=variances,coords=coords,normalize_coords=normalize_coords).call(conv8_2_mbox_loc)

	conv9_2_norm_mbox_priorbox = AnchorBoxes(img_height=img_height,img_width=img_width,this_scale=scales[5],next_scale=scales[6],aspect_ratios=aspect_ratios[5],
	                                 two_boxes_for_ar1=two_boxes_for_ar1,this_steps=steps[5],this_offsets=offsets[5],clip_boxes=clip_boxes,
	                                 variances=variances,coords=coords,normalize_coords=normalize_coords).call(conv9_2_mbox_loc)

	### Reshape

	# Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
	# We want the classes isolated in the last axis to perform softmax on them

	conv4_3_norm_mbox_conf_reshape = tf.reshape(conv4_3_norm_mbox_conf,[-1,n_classes])
	fc7_mbox_conf_reshape = tf.reshape(fc7_mbox_conf,[-1,n_classes])
	conv6_2_mbox_conf_reshape = tf.reshape(conv6_2_mbox_conf,[-1,n_classes])
	conv7_2_mbox_conf_reshape = tf.reshape(conv7_2_mbox_conf,[-1,n_classes])
	conv8_2_mbox_conf_reshape = tf.reshape(conv8_2_mbox_conf,[-1,n_classes])
	conv9_2_mbox_conf_reshape = tf.reshape(conv9_2_mbox_conf,[-1,n_classes])

	# Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
	# We want the four box coordinates isolated in the last axis to compute the smooth L1 loss

	conv4_3_norm_mbox_loc_reshape = tf.reshape(conv4_3_norm_mbox_loc,[-1,4])
	fc7_mbox_loc_reshape = tf.reshape(fc7_mbox_loc,[-1,4])
	conv6_2_mbox_loc_reshape = tf.reshape(conv6_2_mbox_loc,[-1,4])
	conv7_2_mbox_loc_reshape = tf.reshape(conv7_2_mbox_loc,[-1,4])
	conv8_2_mbox_loc_reshape = tf.reshape(conv8_2_mbox_loc,[-1,4])
	conv9_2_mbox_loc_reshape = tf.reshape(conv9_2_mbox_loc,[-1,4])

	# Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`

	conv4_3_norm_mbox_priorbox_reshape = tf.reshape(conv4_3_norm_mbox_priorbox,[-1,8])
	fc7_mbox_priorbox_reshape = tf.reshape(fc7_mbox_priorbox,[-1,8])
	conv6_2_norm_mbox_priorbox_reshape = tf.reshape(conv6_2_norm_mbox_priorbox,[-1,8])
	conv7_2_norm_mbox_priorbox_reshape = tf.reshape(conv7_2_norm_mbox_priorbox,[-1,8])
	conv8_2_norm_mbox_priorbox_reshape = tf.reshape(conv8_2_norm_mbox_priorbox,[-1,8])
	conv9_2_norm_mbox_priorbox_reshape = tf.reshape(conv9_2_norm_mbox_priorbox,[-1,8])

	### Concatenate the predictions from the different layers

	# Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
	# so we want to concatenate along axis 1, the number of boxes per layer

	# Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)

	print(conv4_3_norm_mbox_conf_reshape.get_shape())
	print(fc7_mbox_conf_reshape.get_shape())
	print(conv6_2_mbox_conf_reshape.get_shape())
	print(conv7_2_mbox_conf_reshape.get_shape())
	print(conv8_2_mbox_conf_reshape.get_shape())
	print(conv9_2_mbox_conf_reshape.get_shape())


	mbox_conf = tf.concat([conv4_3_norm_mbox_conf_reshape,
	                       fc7_mbox_conf_reshape,
	                       conv6_2_mbox_conf_reshape,
	                       conv7_2_mbox_conf_reshape,
	                       conv8_2_mbox_conf_reshape,
	                       conv9_2_mbox_conf_reshape],axis=0)

	# Output shape of `mbox_loc`: (batch, n_boxes_total, 4)

	mbox_loc = tf.concat([conv4_3_norm_mbox_loc_reshape,
	                      fc7_mbox_loc_reshape,
	                      conv6_2_mbox_loc_reshape,
	                      conv7_2_mbox_loc_reshape,
	                      conv8_2_mbox_loc_reshape,
	                      conv9_2_mbox_loc_reshape],axis=0)

	# Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)

	mbox_priorbox = tf.concat([conv4_3_norm_mbox_priorbox_reshape,
	                           fc7_mbox_priorbox_reshape,
	                           conv6_2_norm_mbox_priorbox_reshape,
	                           conv7_2_norm_mbox_priorbox_reshape,
	                           conv8_2_norm_mbox_priorbox_reshape,
	                           conv9_2_norm_mbox_priorbox_reshape],axis=0)

	# The box coordinate predictions will go into the loss function just the way they are,
	# but for the class predictions, we'll apply a softmax activation layer first

	mbox_conf_softmax = tf.nn.softmax(mbox_conf)

	# Concatenate the class and box predictions and the anchors to one large predictions vector
	# Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)

	predictions = tf.concat([mbox_conf_softmax, mbox_loc, mbox_priorbox],axis=-1)

	print(predictions.get_shape())



