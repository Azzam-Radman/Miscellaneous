def train_step(optimizer):
    @tf.function
    def one_train_step(adv_x, y, diff, is_adv):
        with tf.GradientTape() as tape:
            pred = model(adv_x, training=True)
            loss = loss_object(y, pred)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if is_adv:
            # subtract differences from the original weights
            restored_weights = [w1 - w2 for w1, w2 in zip(model.trainable_weights, diff)]
            # assign the subtracted weights to the model
            [model.trainable_weights[i].assign(restored_weights[i]) for i in range(len(restored_weights))]

        return loss
    return one_train_step
  
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
training_step = train_step(optimizer) # you can now change the optimizer in each iteration and define variables on each call for the train_step function
for batch, (x1, x2) in tqdm(enumerate(train_ds), total=len(train_ds)):
    total_losses.append(training_step(x1, x2, diff=None, is_adv=False))
