def kl_reconstruction_loss(inputs, ouptuts, mu, sigma):

  kl_loss = 1 + sigma - tf.square(mu) - tf.exp(sigma)
  kl_loss =  - tf.reduce_mean(kl_loss) * 0.5
  return kl_loss

model = tf.keras.Model(inputs=inputs, outputs=outputs)
loss = kl_reconstruction_loss(inputs, z, mu, sigma)
model.add_loss(loss)
