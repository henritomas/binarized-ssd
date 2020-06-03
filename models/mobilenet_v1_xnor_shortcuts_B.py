import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation, PReLU

#Larq layers
import larq as lq
from larq.layers import QuantConv2D, QuantDepthwiseConv2D

#implementation of custom quantizer found here:
#https://github.com/larq/zoo/blob/master/larq_zoo/literature/xnornet.py
@lq.utils.set_precision(1)
@lq.utils.register_keras_custom_object
def xnor_weight_scale(x):
    """
    Clips the weights between -1 and +1 and then calculates a scale factor per
    weight filter. See https://arxiv.org/abs/1603.05279 for more details
    """
    x = tf.clip_by_value(x, -1, 1)
    alpha = tf.reduce_mean(tf.abs(x), axis=[0, 1, 2], keepdims=True)
    return alpha * lq.quantizers.ste_sign(x)

#Stage 1: kernel quantizer and constraint = None (binary activations, real weights)
stage_1 = dict(input_quantizer="ste_sign",
          kernel_quantizer=None,
          kernel_constraint=None)

#Stage 2: kernel quantizer and constraint are set (binary everything)
stage_2 = dict(input_quantizer="ste_sign",
          kernel_quantizer="xnor_weight_scale",
          kernel_constraint="weight_clip")

#Depthwise convolutions must always be full precision
#Too much information loss when dw are binarized
d_kwargs = dict(input_quantizer=None,
                depthwise_quantizer=None,
                depthwise_constraint=None)

#kwargs = stage_2

#First layer only, not quantized
def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), use_prelu=False):

    channel_axis = 3 #last index is channels
    filters = int(filters * alpha)

    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(axis=channel_axis, momentum=0.99, epsilon=0.001, name='conv1_bn')(x)

    if use_prelu:
        x = PReLU(shared_axes=[1,2], name='conv1_prelu')(x)
    else:
        x = Activation('relu', name='conv1_relu')(x)

    return x

#Succeeding layers, are quantized
def _depthwise_conv_block_classification(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1, 
                          stage=2, binary_ds=False, use_prelu=False):
 
    channel_axis = 3
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if not binary_ds and strides==(2,2):
        p_kwargs = dict(input_quantizer=None,
                        kernel_quantizer=None,
                        kernel_constraint=None)
    elif stage==1:
        p_kwargs = stage_1
    elif stage==2:
        p_kwargs = stage_2
    else:
        raise ValueError("Stage must be specified with ints 1 or 2.")

    x = QuantDepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id,
                        **d_kwargs)(inputs)
    if use_prelu:
        x = PReLU(shared_axes=[1,2], name='conv_dw_%d_prelu' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.99, epsilon=0.001, name='conv_dw_%d_bn' % block_id)(x)
    sc = x

    x = QuantConv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id,
               **p_kwargs)(x)
    if use_prelu:
        x = PReLU(shared_axes=[1,2], name='conv_pw_%d_prelu' % block_id)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.99, epsilon=0.001, name='conv_pw_%d_bn' % block_id)(x)
    
    if not strides==(2,2) or binary_ds:
        x = Add()([x, sc])

    return x

def mobilenet(input_tensor, alpha=1.0, depth_multiplier=1, stage=2, binary_ds=False, use_prelu=False):

    train_args = dict(
        stage=stage,
        binary_ds=binary_ds,
        use_prelu=use_prelu
    )

    if input_tensor is None:
        input_tensor = Input(shape=(300,300,3))

    x = _conv_block(input_tensor, 32, alpha, strides=(2, 2), use_prelu=use_prelu)
    x = _depthwise_conv_block_classification(x, 64, alpha, depth_multiplier, block_id=1, **train_args)

    x = _depthwise_conv_block_classification(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2, **train_args)
    x = _depthwise_conv_block_classification(x, 128, alpha, depth_multiplier, block_id=3, **train_args)

    x = _depthwise_conv_block_classification(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4, **train_args)
    x = _depthwise_conv_block_classification(x, 256, alpha, depth_multiplier, block_id=5, **train_args)

    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6, **train_args)
    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=7, **train_args)
    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=8, **train_args)
    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=9, **train_args)
    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=10, **train_args)
    conv4_3 = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=11, **train_args) #11 conv4_3 (300x300)-> 19x19 

    x = _depthwise_conv_block_classification(conv4_3, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12, **train_args)   # (300x300) -> 10x10 
    fc7 = _depthwise_conv_block_classification(x, 1024, alpha, depth_multiplier, block_id=13, **train_args) # 13 fc7 (300x300) -> 10x10

    #model = Model(inputs=input_tensor, outputs=fc7)
    #return model

    return [conv4_3, fc7]