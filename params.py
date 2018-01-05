class TrainingParams:
    def __init__(self):
        self.train_path='data/train/val2017/*.jpg'
        self.style_path='data/starry_night.jpg'
        self.content_weight = 1.0
        self.style_weight = 5.0
        self.tv_weight = 0.0001
        self.batch_size = 1
        self.input_shape = [256,256,3]
        self.num_epoch = 2
        self.learn_rate = 0.001
        self.total_train_sample = 4
        self.save_path = 'summaries'
        self.summary_step = 100
        self.read_thread = 1

class TransferParams:
    def __init__(self):
        self.input_shape = [256,256,3]
        self.save_path = 'summaries'
        