"""Lite R-ASPP Semantic Segmentation based on MobileNetV3.
"""

from mobilenet_v3_small import MobileNetV3_Small
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, BatchNormalization, Activation, Multiply, Add, Reshape
from tensorflow.keras.utils import plot_model
from bilinear_upsampling import BilinearUpSampling2D
from tensorflow.image import ResizeMethod
import tensorflow as tf


class LiteRASSP:
    def __init__(self, input_shape, n_class=2, alpha=1.0, weights=None):
        """Init.
        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor (should be 1024 × 2048 or 512 × 1024 according 
                to the paper).
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier for mobilenetV3.
            weights: String, weights for mobilenetv3.
            backbone: String, name of backbone (must be small or large).
        """
        self.size = input_shape[0]
        self.shape = input_shape
        self.n_class = n_class
        self.alpha = alpha
        self.weights = weights

    def _extract_backbone(self, plot=False):
        """extract feature map from backbone.
        """
        model = MobileNetV3_Small(self.shape, self.n_class, alpha=self.alpha, include_top=False).build(plot=plot)
        layer_name8 = "batch_normalization_7"
        layer_name16 = "add_2"
        if self.weights is not None:
            model.load_weights(self.weights, by_name=True)

        inputs= model.input
        # 1/8 feature map.
        out_feature8 = model.get_layer(layer_name8).output
        # 1/16 feature map.
        out_feature16 = model.get_layer(layer_name16).output

        return inputs, out_feature8, out_feature16

    def build(self, plot=False):
        """build Lite R-ASPP.
        # Arguments
            plot: Boolean, weather to plot model.
        # Returns
            model: Model, model.
        """
        inputs, out_feature8, out_feature16 = self._extract_backbone(plot=plot)

        # branch1
        x1 = Conv2D(128, (1, 1))(out_feature16)
        x1 = BatchNormalization()(x1)
        x1 = Activation("relu")(x1)

        # branch2
        s = x1.shape

        x2 = AveragePooling2D(pool_size=(25, 25), strides=(8, 8))(out_feature16)
        x2 = Conv2D(128, (1, 1))(x2)
        x2 = Activation("sigmoid")(x2)
        x2 = BilinearUpSampling2D(target_size=(int(s[1]), int(s[2])))(x2)

        # branch3
        x3 = Conv2D(self.n_class, (1, 1))(out_feature8)

        # merge1
        x = Multiply()([x1, x2])
        x = BilinearUpSampling2D(size=(2, 2))(x)
        x = Conv2D(self.n_class, (1, 1))(x)

        # merge2
        x = Add()([x, x3])

        # out
        o = tf.image.resize(x,
                            size=(self.size, self.size),
                            method=ResizeMethod.BILINEAR,
                            preserve_aspect_ratio=False,
                            antialias=False,
                            name=None)
        o = (Reshape((self.size * self.size, -1)))(o)
        o = Activation("softmax")(o)

        model = Model(inputs=inputs, outputs=o)

        if plot:
            plot_model(model, to_file="images/LR_ASPP.png", show_shapes=True)

        return model