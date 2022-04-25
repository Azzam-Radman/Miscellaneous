lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch/20)) # epoch strats from 1 (gradually increase LR to 1e-3)
history = model.fit(train_set, epochs=100, callbacks=[lr_scheduler]) # train for 100 epoch then make a plot, visualize the best performing LR and pick it

# then plot the loss versus the learning rate
plt.semilogx(history.history['lr'], history.history['loss'])
plt.axis([1e-8, 1e-4, 0, 30])
