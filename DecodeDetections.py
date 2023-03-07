from __future__ import division
import numpy as np 
import tensorflow as tf 


class DecodeDetections(Layer):

	'''TensorFlow layer which we have created in model_fn.py file ,

	while training a model.

	Layer is a 3D Tensor with shape (batch_size,n_boxes,n_classes +12)

	12 last magnitudes consists of 4 predicted shifts of anchors (cx,cy,w,h)

	and of 8 default anchor boxes  coordinates (cx,cy,w,h,var1,var1,var2,var2)

	Layer= confidences + locations + anchors

	INPUT_SHAPE : (batch_size,n_boxes,n_classes+12)

	OUTPUT SHAPE : (batch_size,top_k,6)

	'''

	def __init__(self,
		confidence_thresh=0.01,
		iou_threshold=0.45,
		top_k=200,
		nms_max_output_size=400,
		coords='centroids',
		normalize_coords=True,
		img_height=None,
		img_width=None,
		**kwargs):

	'''
	All default argument values follow the Caffe implementation.
	Arguments:

	confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
	positive class in order to be considered for the non-maximum suppression stage for the respective class.
	A lower value will result in a larger part of the selection process being done by the non-maximum suppression
	stage, while a larger value will result in a larger part of the selection process happening in the confidence
	thresholding stage.

	iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than iou_threshold
	with a locally maximal box will be removed from the set of predictions for a given class, where maximal refers
	to the box score.

	top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
	non-maximum suppression stage.

	nms_max_output_size (int, optional): The maximum number of predictions that will be left after performing non-maximum
	suppression.

	coords (str, optional): The box coordinate format that the model outputs. Must be 'centroids'
	i.e. the format (cx, cy, w, h) (box center coordinates, width, and height). Other coordinate formats are
	currently not supported.

	normalize_coords (bool, optional): Set to True if the model outputs relative coordinates (i.e. coordinates in [0,1])
	and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
	relative coordinates, but you do not want to convert them back to absolute coordinates, set this to False.
	Do not set this to True if the model already outputs absolute coordinates, as that would result in incorrect
	coordinates. Requires img_height and img_width if set to True.

	img_height (int, optional): The height of the input images. Only needed if normalize_coords is True.

	img_width (int, optional): The width of the input images. Only needed if normalize_coords is True.

	'''


	# We need these members for the config.
	self.confidence_thresh = confidence_thresh
	self.iou_threshold = iou_threshold
	self.top_k = top_k
	self.normalize_coords = normalize_coords
	self.img_height = img_height
	self.img_width = img_width
	self.coords = coords
	self.nms_max_output_size = nms_max_output_size

	
	# We need these members for TensorFlow.
	self.tf_confidence_thresh = tf.constant(self.confidence_thresh, name='confidence_thresh')
	self.tf_iou_threshold = tf.constant(self.iou_threshold, name='iou_threshold')
	self.tf_top_k = tf.constant(self.top_k, name='top_k')
	self.tf_normalize_coords = tf.constant(self.normalize_coords, name='normalize_coords')
	self.tf_img_height = tf.constant(self.img_height, dtype=tf.float32, name='img_height')
	self.tf_img_width = tf.constant(self.img_width, dtype=tf.float32, name='img_width')
	self.tf_nms_max_output_size = tf.constant(self.nms_max_output_size, name='nms_max_output_size')


	# super(DecodeDetections, self).__init__(**kwargs) #I DONT UNDERSTAND WHAT IT MEANS

	# def build(self, input_shape):
	# 	self.input_spec = [InputSpec(shape=input_shape)]
	# 	super(DecodeDetections, self).build(input_shape)


	def call(self,y_pred,mask=None):

	'''
	Returns:
	3D tensor of shape (batch_size, top_k, 6). The second axis is zero-padded
	to always yield `top_k` predictions per batch item. The last axis contains
	the coordinates for each predicted box in the format
	[class_id, confidence, xmin, ymin, xmax, ymax].
	'''

	###########################################################################
	'''STAGE 1. Convert the box coordinates from predicted anchor 
	box offsets to predictedabsolute coordinates'''
	###########################################################################

	#Convert anchor box offsets to image offsets.

	# cx = cx_pred * cx_variance * w_anchor + cx_anchor
	cx = y_pred[...,-12] * y_pred[...,-4] * y_pred[...,-6] + y_pred[...,-8]
	# cy = cy_pred * cy_variance * h_anchor + cy_anchor
	cy = y_pred[...,-11] * y_pred[...,-3] * y_pred[...,-5] + y_pred[...,-7]
	# w = exp(w_pred * variance_w) * w_anchor
	w = tf.exp(y_pred[...,-10] * y_pred[...,-2]) * y_pred[...,-6]
	# h = exp(h_pred * variance_h) * h_anchor
	h = tf.exp(y_pred[...,-9] * y_pred[...,-1]) * y_pred[...,-5]

	#Convert 'centroids' to corners

	xmin = cx - 0.5 * w
	ymin = cy - 0.5 * h
	xmax = cx + 0.5 * w
	ymax = cy + 0.5 * h

	#We can choose of of two types of coords. Normalized ccords and Non Normalized

	#Values of xmin,ymin,xmax,ymax lie between 0 and 1
	def normalized_coords():
		xmin1 = tf.expand_dims(xmin * self.tf_img_width, axis=-1)
		ymin1 = tf.expand_dims(ymin * self.tf_img_height, axis=-1)
		xmax1 = tf.expand_dims(xmax * self.tf_img_width, axis=-1)
		ymax1 = tf.expand_dims(ymax * self.tf_img_height, axis=-1)
		return xmin1, ymin1, xmax1, ymax1

	#Values of xmin,xmax lie between 0 and img_width and ymin,ymax lie between 0 and img_height
	def non_normalized_coords():
		return tf.expand_dims(xmin, axis=-1), tf.expand_dims(ymin, axis=-1), 
		tf.expand_dims(xmax, axis=-1), tf.expand_dims(ymax, axis=-1)


	xmin, ymin, xmax, ymax = tf.cond(self.tf_normalize_coords, normalized_coords, non_normalized_coords)

	
	'''Concatenate the one-hot class confidences and the converted box coordinates 
	to form the decoded predictions tensor'''

	#y_pred shape (batch_size,n_boxes,n_classes+4)
	y_pred = tf.concat(values=[y_pred[...,:-12], xmin, ymin, xmax, ymax], axis=-1)


	###########################################################################
	''' STAGE 2. Perform :
	-confidence thresholding
	-per-class non-maximum suppression
	-top-k filtering'''
	###########################################################################

	batch_size=tf.shape(y_pred)[0] #Output dtype: tf.int32
	n_boxes=tf.shape(y_pred)[1]
	n_classes=y_pred[2]-4
	class_indices=tf.range(1,n_classes)

	# Create a function that filters the predictions for the given batch item. Specifically, it performs:
	# - confidence thresholding
	# - non-maximum suppression (NMS)
	# - top-k filtering


	#Here,we iterate by batch_size.And shape becomes (n_boxes,classes+4)
	def filter_predictions(batch_item):

		# Create a function that filters the predictions for one single class.
		def filter_class(index)