import tensorflow as tf
from tensorflow.keras import Model
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


class DepthWise_PointWise_Layer(Layer):
    def __init__(self, out_channels=64, dw_stride=1, padding="same", **kwargs):
        super().__init__(**kwargs)

        self.depthwise_stride = dw_stride
        self.num_out_channels = out_channels

        self.depthwise_conv = DepthwiseConv2D(
            kernel_size=3,
            strides=self.depthwise_stride,
            padding=padding,
            use_bias=False,
        )

        self.pointwise_conv = Conv2D(
            filters=self.num_out_channels,
            kernel_size=1,
            strides=1,
            padding=padding,
            use_bias=False,
        )

        self.bn_1 = BatchNormalization()
        self.bn_2 = BatchNormalization()

        self.relu = ReLU(max_value=6.0)

    def call(self, data, training=True, **kwargs):

        out = self.depthwise_conv(data)
        out = self.relu(self.bn_1(out, training=training))

        out = self.pointwise_conv(out)
        out = self.relu(self.bn_2(out, training=training))

        return out


def create_mobilenet_v1(
    input_shape=(224, 224, 3),
    alpha=1.0,
    num_classes=1000,
    pooling="avg",
    dropout_rate=0.3,
    use_dense=True,
):

    all_out_channels = [64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
    all_DW_strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]

    pooling_keep_dims = False if use_dense else True
    pooling_layer = getattr(tf.keras.layers, f"Global{pooling.capitalize()}Pooling2D")(keepdims=pooling_keep_dims)

    input_layer = Input(shape=input_shape)

    out = Conv2D(
        filters=int(32 * alpha),
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name="Conv_0",
    )(input_layer)

    out = BatchNormalization(name="BN_0")(out)
    out = ReLU(max_value=6.0, name="ReLU_0")(out)

    for block_id, (out_channels, dw_stride) in enumerate(zip(all_out_channels, all_DW_strides), 1):

        num_out_channels = int(out_channels * alpha)

        out = DepthWise_PointWise_Layer(
            out_channels=num_out_channels,
            dw_stride=dw_stride,
            name=f"DW_PW__{block_id:02}",
        )(out)

    pool = pooling_layer(out)
    drop_out = Dropout(rate=dropout_rate)(pool)

    if not use_dense:
        conv_out = Conv2D(filters=num_classes, kernel_size=1, strides=1, activation="softmax")(drop_out)
        final = Flatten()(conv_out)
    else:
        final = Dense(units=num_classes, activation="softmax", name="FC")(drop_out)

    mobilenet_v1_model = Model(inputs=input_layer, outputs=final, name="MobileNet-V1")

    return mobilenet_v1_model


if __name__ == "__main__":

    model = create_mobilenet_v1(alpha=1.0, pooling="average", use_dense=False)
    model.summary()

    # for alpha in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
    #     model = create_mobilenet_v1(alpha=alpha, pooling="average", use_dense=False)
    #     print(f"Alpha: {alpha} --->", model.count_params())
