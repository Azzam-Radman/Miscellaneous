def step_decay_wrapper(initial_lr, num_epochs, args):
    def step_decay(epoch):
        decays = np.linspace(0, np.pi/2, num_epochs//4)
        decay_epoch = int(epoch % (num_epochs/4))
        drop = decays[decay_epoch]
        cos_drop = max(np.cos(drop), 1e-5)
        lr = initial_lr * cos_drop
        return lr
    return step_decay(*args)
  
lrs = []
for i in range(100):
    lrs.append(step_decay_wrapper(0.01, 100, [i]))
