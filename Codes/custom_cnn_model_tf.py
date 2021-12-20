class Block(tf.keras.Model):
    def __init__(self, filters, kernel_size, kernel_strides, repetitions, pool_size, pool_strides, padding='valid', 
                 dropout=False, dropout_ratio=None):
        super(Block, self).__init__()
        self.repetitions = repetitions
        self.dropout = dropout
        self.dropout_ratio = dropout_ratio
        
        for i in range(repetitions):
            vars(self)[f'conv_{i}'] = Conv2D(filters=filters, kernel_size=kernel_size, 
                                             strides=kernel_strides, padding=padding, 
                                             activation='relu')
            vars(self)[f'bn_{i}'] = BatchNormalization()
            vars(self)[f'drop_{i}'] = Dropout(dropout_ratio)
            
        self.max_pool = MaxPooling2D(pool_size=pool_size, strides=pool_strides)
        
    def call(self, inputs):
        x = inputs
        for i in range(self.repetitions):
            x = vars(self)[f'conv_{i}'](x)
            x = vars(self)[f'bn_{i}'](x)
            if self.dropout:
                x = vars(self)[f'drop_{i}'](x)
            
        max_pool = self.max_pool(x)
        return max_pool
      
class CustomModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.block_0 = Block(64, 3, 2, 3, 2, 2, padding='valid', dropout=True, dropout_ratio=0.3)
        self.flatten = Flatten()
        self.fc = Dense(num_classes, activation='softmax')
        
    def call(self, inputs):
        x = self.block_0(inputs)
        x = self.flatten(x)
        output = self.fc(x)
        return output
      
mg_adjust_layer = tf.keras.layers.Lambda(lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="torch"), 
                                              input_shape=[224, 224, 3])
    
base_model = CustomModel(10)
model = tf.keras.models.Sequential([img_adjust_layer, base_model])
