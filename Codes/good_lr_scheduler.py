lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**((epoch+1)/20)) # epoch strats from 0 (gradually increase LR to 1e-3)
history = model.fit(train_set, epochs=100, callbacks=[lr_scheduler]) # train for 100 epoch then make a plot, visualize the best performing LR and pick it

# then plot the loss versus the learning rate
plt.figure(figsize=(12, 7))
plt.semilogx(history.history['lr'], history.history['val_mae'])
min_lr = history.history['lr'].min()
max_lr = history.history['lr'].max()
min_mae = history.history['val_mae'].min()
max_mae = history.history['val_mae'].max()
plt.axis([min_lr, max_lr, min_mae, max_mae])
plt.show()
