class CnnBlock(tf.keras.Model):
    def __init__(self, repetitions, filters, kernel_size, kernel_strides, pool_size, pool_strides, padding='valid'):
        super(CnnBlock, self).__init__()
        self.repetitions = repetitions
        
        for i in range(repetitions):
            vars(self)[f'cnn_{i}'] = Conv2D(filters=filters, kernel_size=kernel_size, 
                                            padding=padding, strides=kernel_strides, activation='relu')
            vars(self)[f'bn_{i}'] = BatchNormalization()
        
        self.max_pool = MaxPooling2D(pool_size=pool_size, strides=pool_strides)
        
    def call(self, inputs):
        x = inputs
        for i in range(self.repetitions):
            conv_i = vars(self)[f'cnn_{i}']
            x = conv_i(x)
            bn_i = vars(self)[f'bn_{i}']
            x = bn_i(x)
        max_pool = self.max_pool(x)
        return max_pool
      
      
class CustomModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.block_0 = CnnBlock(1, 3, 11, 2, 2, 2, 'valid')
        self.flatten = Flatten()
        self.fc = Dense(128, activation='relu')
        self.classifier = Dense(num_classes, activation='softmax')
        
    def call(self, inputs):
        x = self.block_0(inputs)
        x = self.flatten(x)
        x = self.fc(x)
        output = self.classifier(x)
        return output

# very necessary to adjust input size
img_adjust_layer = tf.keras.layers.Lambda(lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="torch"), 
                                              input_shape=[224, 224, 3])
model_ = CustomModel(10)
model = tf.keras.models.Sequential([
    img_adjust_layer,
    model_
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
            )

model.fit(train_ds, validation_data=valid_ds, epochs=100, callbacks=[lr])
