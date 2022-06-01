import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dense, Multiply


class SqeezeExcitation(Layer):
    def __init__(self, num_channels=64, reduction_factor=16, use_bias=False, **kwargs):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.reduction_factor = reduction_factor

        self.reduced_out_channels = self.num_channels // self.reduction_factor

        self.scaling = Multiply()

        self.global_avg_pooling = GlobalAveragePooling2D(name="squeeze")
        self.dimension_reduction = Dense(self.reduced_out_channels, activation="relu", use_bias=use_bias)
        self.dimension_expansion = Dense(self.num_channels, activation="sigmoid", use_bias=use_bias)

    def call(self, features):

        # Squeeze
        x = self.global_avg_pooling(features)

        # Excite
        x = self.dimension_reduction(x)
        x = self.dimension_expansion(x)

        # Scaling
        x = self.scaling([features, x])

        return x


if __name__ == "__main__":
    num_channels = 64
    reduce_factor = 32

    se = SqeezeExcitation(num_channels=num_channels, reduction_factor=reduce_factor)

    print(se(tf.random.normal((1, 10, 10, num_channels))).shape)
