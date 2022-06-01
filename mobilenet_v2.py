import tensorflow as tf
from tensorflow.keras import Sequential, Model
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
)


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidualBlock(Layer):
    def __init__(
        self,
        in_channels=32,
        out_channels=64,
        depthwise_stride=1,
        expansion_channels=32,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Input Parameters
        self.num_in_channels = in_channels
        self.num_out_channels = out_channels
        self.depthwise_stride = depthwise_stride
        self.expansion_channels = expansion_channels

        # Layer Attributes
        self.apply_expansion = self.expansion_channels > self.num_in_channels
        self.residual_connection = True if (self.num_in_channels == self.num_out_channels) and (self.depthwise_stride == 1) else False

        # Layers
        self.sequential_block = Sequential()

        if self.apply_expansion:
            self.sequential_block.add(Conv2D(filters=self.expansion_channels, kernel_size=1, strides=1, use_bias=False))
            self.sequential_block.add(BatchNormalization())
            self.sequential_block.add(ReLU(max_value=6.0))

        self.sequential_block.add(DepthwiseConv2D(kernel_size=3, strides=self.depthwise_stride, padding="same", use_bias=False))
        self.sequential_block.add(BatchNormalization())
        self.sequential_block.add(ReLU(max_value=6.0))

        self.sequential_block.add(Conv2D(filters=self.num_out_channels, kernel_size=1, strides=1, use_bias=False))
        self.sequential_block.add(BatchNormalization())
        self.sequential_block.add(ReLU(max_value=6.0))

    def call(self, data, **kwargs):

        out = self.sequential_block(data)

        if self.residual_connection:
            out = out + data

        return out


def create_mobilenet_v2(
    input_shape=(224, 224, 3),
    alpha=1.0,
    num_classes=1000,
    pooling="average",
    dropout_rate=0.3,
    use_dense=True,
):

    pooling_keep_dims = False if use_dense else True
    pooling_layer = getattr(tf.keras.layers, f"Global{pooling.capitalize()}Pooling2D")(keepdims=pooling_keep_dims)

    input_channels = int(_make_divisible(32 * alpha))

    # t = expansion rate
    # c = number of output_channels
    # n = number of block repeats
    # s = strides

    bottleneck_stats = (
        # t,  c,  n, s
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    )

    input_layer = Input(shape=input_shape)

    out = Conv2D(
        filters=input_channels,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name="Conv_0",
    )(input_layer)
    out = BatchNormalization(name="BN_0")(out)
    out = ReLU(max_value=6.0, name="ReLU_0")(out)

    for block_main_id, bottleneck in enumerate(bottleneck_stats, 1):

        t = bottleneck[0]  # expansion rate
        c = bottleneck[1]  # number of output_channels
        n = bottleneck[2]  # number of block repeats
        s = bottleneck[3]  # Stride of first block in n blocks

        for block_repeat_num in range(1, n + 1):
            # Alpha updated channels
            out_channels = int(_make_divisible(c * alpha))
            expansion_channels = int(_make_divisible(t * input_channels))

            out = InvertedResidualBlock(
                in_channels=input_channels,
                expansion_channels=expansion_channels,
                out_channels=out_channels,
                depthwise_stride=s,
                name=f"IR__{block_main_id:02}_{block_repeat_num:02}",
            )(out)

            s = 1  # stride for remaining blocks is 1.
            input_channels = out_channels  # number of input_channels for the next block.

    conv_final_filters = int(_make_divisible(1280 * alpha)) if alpha > 1.0 else 1280

    out = Conv2D(filters=conv_final_filters, kernel_size=1, strides=1, use_bias=False, name="Conv_final")(out)
    out = BatchNormalization(name="BN_final")(out)
    out = ReLU(max_value=6.0, name="ReLU_final")(out)

    pool = pooling_layer(out)
    drop_out = Dropout(rate=dropout_rate)(pool)

    if not use_dense:
        conv_out = Conv2D(filters=num_classes, kernel_size=1, strides=1, activation="softmax", name="Conv_out")(drop_out)
        final = Flatten()(conv_out)
    else:
        final = Dense(units=num_classes, activation="softmax", name="Dense_out")(drop_out)

    mobilenet_v2_model = Model(inputs=input_layer, outputs=final, name="MobileNet-V2")

    return mobilenet_v2_model


if __name__ == "__main__":

    alpha = 1.0
    model = create_mobilenet_v2(alpha=alpha)
    print(f"Alpha: {alpha} --->", model.count_params())

    # for alpha in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
    #     model = create_mobilenet_v2(alpha=alpha, pooling="average", use_dense=True)
    #     print(f"Alpha: {alpha} --->", model.count_params())
