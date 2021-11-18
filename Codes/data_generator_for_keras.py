def generator(X, y, batch_size=32):
    
    num_samples=len(X)
    
    while True:
        for offset in range(0, num_samples, batch_size):
            x_train = X[offset:offset+batch_size]
            y_train = y[offset:offset+batch_size]

            yield x_train, y_train
            
BATCH_SIZE = 256
EPOCHS=20

train_gen = generator(X_train, y_train, batch_size=BATCH_SIZE)
valid_gen = generator(X_valid, y_valid, batch_size=BATCH_SIZE)

model.fit(
    x=train_gen,
    steps_per_epoch=len(X_train)//BS,
    validation_data=valid_gen,
    validation_steps=len(X_valid)//BS,
    epochs=EPOCHS,
)
