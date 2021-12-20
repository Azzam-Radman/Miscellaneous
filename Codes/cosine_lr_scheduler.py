def step_decay_wrapper(initial_lr, num_epochs, min_lr):
    def step_decay(epoch):
        decays = np.linspace(0, np.pi/2, num_epochs//4)
        decay_epoch = int(epoch % (num_epochs/4))
        drop = decays[decay_epoch]
        cos_drop = np.cos(drop)
        lr = max(initial_lr * cos_drop, min_lr)
        return lr
    return step_decay
  
  LR_scheduler = LearningRateScheduler(step_decay_wrapper(LR, EPOCHS, 1e-6), verbose=1)
