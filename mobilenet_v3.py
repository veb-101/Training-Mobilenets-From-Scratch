import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.activations import hard_sigmoid

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Layer,
    Input,
    BatchNormalization,
    ReLU,
    Conv2D,
    DepthwiseConv2D,
    Dense,
    Flatten,
    Dropout,
    GlobalAveragePooling2D,
    Dense,
    Multiply,
)


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqeezeExcitation(Layer):
    def __init__(self, num_channels=64, use_bias=False, **kwargs):
        super().__init__(**kwargs)

        # Layer attributes
        self.num_channels = num_channels
        self.reduced_out_channels = _make_divisible(self.num_channels // 4)

        # Layers
        self.global_avg_pooling = GlobalAveragePooling2D(keepdims=True, name="squeeze")
        self.dimension_reduction = Conv2D(filters=self.reduced_out_channels, kernel_size=1, activation="relu", use_bias=use_bias)
        self.dimension_expansion = Conv2D(filters=self.num_channels, kernel_size=1, use_bias=use_bias)

        self.scaling = Multiply()

    def call(self, features):

        # Squeeze
        x = self.global_avg_pooling(features)

        # Excite
        x = self.dimension_reduction(x)
        x = self.dimension_expansion(x)
        x = hard_sigmoid(x)

        # Scaling
        x = self.scaling([features, x])

        return x


class HardSwish(Layer):
    def call(self, data):
        return data * tf.nn.relu6(data + 3) * 0.16666667


class InvertedResidualBlock(Layer):
    def __init__(
        self,
        kernel_size=3,
        in_channels=32,
        out_channels=64,
        depthwise_stride=1,
        activation_fn="RE",
        expansion_size=32,
        apply_SE=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Input Parameters
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.expansion_size = expansion_size
        self.num_out_channels = out_channels
        self.depthwise_stride = depthwise_stride
        self.activation_string = activation_fn
        self.apply_SE = apply_SE

        # Layer Attributes
        self.apply_expansion = self.expansion_size != self.in_channels
        self.activation_fn = ReLU if self.activation_string == "RE" else HardSwish
        self.residual_connection = True if (self.in_channels == self.num_out_channels) and (self.depthwise_stride == 1) else False

        # ====================================================================================================================
        # # ================================================== Build Layers ==================================================
        # ====================================================================================================================
        self.sequential_block = Sequential()

        if self.apply_expansion:
            self.sequential_block.add(Conv2D(filters=self.expansion_size, kernel_size=1, strides=1, use_bias=False))
            self.sequential_block.add(BatchNormalization())
            self.sequential_block.add(self.activation_fn())

        self.sequential_block.add(DepthwiseConv2D(kernel_size=self.kernel_size, strides=self.depthwise_stride, padding="same", use_bias=False))
        self.sequential_block.add(BatchNormalization())
        self.sequential_block.add(self.activation_fn())

        if self.apply_SE:
            self.sequential_block.add(SqeezeExcitation(num_channels=self.expansion_size, use_bias=True))

        self.sequential_block.add(Conv2D(filters=self.num_out_channels, kernel_size=1, strides=1, use_bias=False))
        self.sequential_block.add(BatchNormalization())

    def call(self, data, **kwargs):

        out = self.sequential_block(data)

        if self.residual_connection:
            out = out + data

        return out


def create_mobilenet_v3(
    input_shape=(224, 224, 3),
    alpha=1.0,
    num_classes=1000,
    pooling="average",
    dropout_rate=0.3,
    use_dense=False,
    large=True,
):

    pooling_keep_dims = False if use_dense else True
    pooling_layer = getattr(tf.keras.layers, f"Global{pooling.capitalize()}Pooling2D")(keepdims=pooling_keep_dims)

    layer_1_out_channels = 16

    # k = kernel_size
    # t = expansion size
    # c = number of output channels
    # SE = apply Squeeze and Excitation
    # NL = non-linear function to use, "RE": relu, "HS": hard-swish
    # s = strides

    v3_large_blocks = (
        # k,  t,  c,  SE, NL,  s
        (3, 16, 16, False, "RE", 1),
        (3, 64, 24, False, "RE", 2),
        (3, 72, 24, False, "RE", 1),
        (5, 72, 40, True, "RE", 2),
        (5, 120, 40, True, "RE", 1),
        (5, 120, 40, True, "RE", 1),
        (3, 240, 80, False, "HS", 2),
        (3, 200, 80, False, "HS", 1),
        (3, 184, 80, False, "HS", 1),
        (3, 184, 80, False, "HS", 1),
        (3, 480, 112, True, "HS", 1),
        (3, 672, 112, True, "HS", 1),
        (5, 672, 160, True, "HS", 2),
        (5, 960, 160, True, "HS", 1),
        (5, 960, 160, True, "HS", 1),
    )

    v3_small_blocks = (
        # k,  t,  c,  SE, NL,  s
        (3, 16, 16, True, "RE", 2),
        (3, 72, 24, False, "RE", 2),
        (3, 88, 24, False, "RE", 1),
        (5, 96, 40, True, "HS", 2),
        (5, 240, 40, True, "HS", 1),
        (5, 240, 40, True, "HS", 1),
        (5, 120, 48, True, "HS", 1),
        (5, 144, 48, True, "HS", 1),
        (5, 288, 96, True, "HS", 2),
        (5, 576, 96, True, "HS", 1),
        (5, 576, 96, True, "HS", 1),
    )

    if large:
        v3_blocks = v3_large_blocks
        # last_conv_channel = _make_divisible(960 * alpha)
        last_point_channel = _make_divisible(1280 * alpha) if alpha > 1.0 else 1280
        use_SE = False
    else:
        v3_blocks = v3_small_blocks
        # last_conv_channel = _make_divisible(576 * alpha)
        last_point_channel = _make_divisible(1024 * alpha) if alpha > 1.0 else 1024
        use_SE = False

    # ========================================================================================
    # ================================ MODEL BUILDING ========================================
    # ========================================================================================
    input_layer = Input(shape=input_shape)

    out = Conv2D(filters=layer_1_out_channels, kernel_size=3, strides=2, padding="same", use_bias=False, name="Conv_0")(input_layer)
    out = BatchNormalization(name="BN_0")(out)
    out = HardSwish()(out)

    # Keep record of true #output_channels from the previous layer
    prev_c = layer_1_out_channels

    # Next layer will #layer_1_out_channels as its input_channels
    input_channels = layer_1_out_channels

    for block_main_id, bottleneck in enumerate(v3_blocks, 1):

        k = bottleneck[0]
        t = bottleneck[1]
        c = bottleneck[2]
        SE = bottleneck[3]
        NL = bottleneck[4]
        s = bottleneck[5]

        # alpha updated output_channels
        output_channels = int(_make_divisible(c * alpha))

        # True expansion ratio
        true_expansion_ratio = t / prev_c
        true_expansion_size = _make_divisible(input_channels * true_expansion_ratio)

        out = InvertedResidualBlock(
            kernel_size=k,
            in_channels=input_channels,
            out_channels=output_channels,
            depthwise_stride=s,
            activation_fn=NL,
            expansion_size=true_expansion_size,
            apply_SE=SE,
            name=f"IR__{block_main_id:02}",
        )(out)

        # #input_channels for the next layer
        input_channels = output_channels

        # Keeping track of previous layers original #output_channels
        # to calculate the correct expansion ratio
        prev_c = c

    # ========================================================================================
    last_conv_channel = input_channels * 6

    out = Conv2D(filters=last_conv_channel, kernel_size=1, strides=1, use_bias=False)(out)
    out = BatchNormalization()(out)
    out = HardSwish()(out)

    if use_SE:
        out = SqeezeExcitation(num_channels=last_conv_channel, use_bias=False)(out)
    # ========================================================================================
    out = pooling_layer(out)
    # ========================================================================================
    out = Conv2D(filters=last_point_channel, kernel_size=1, strides=1, use_bias=True)(out)
    out = HardSwish()(out)

    drop_out = Dropout(rate=dropout_rate)(out)

    if not use_dense:
        conv_out = Conv2D(filters=num_classes, kernel_size=1, strides=1, activation="softmax", name="Conv_out")(drop_out)
        final = Flatten()(conv_out)
    else:
        final = Dense(units=num_classes, activation="softmax", name="Dense_out")(drop_out)

    mobilenet_v3_model = Model(inputs=input_layer, outputs=final, name="MobileNet-V3")

    return mobilenet_v3_model


if __name__ == "__main__":

    alpha = 1.0
    model = create_mobilenet_v3(large=False, alpha=alpha)
    print(f"Alpha: {alpha} --->", model.count_params())

    # for alpha in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
    #     model = create_mobilenet_v3(large=True, alpha=alpha)
    #     # model = create_mobilenet_v3(large=False, alpha=alpha)
    #     print(f"Alpha: {alpha} --->", model.count_params())
