class Callback():
    '''
    Callback basic class.

    Callback has the following methods:

    * __init__(self) -- constructor

    * on_train_begin(self) -- method that is executed when training starts
        (in the beginning of ```trainer.train```)

    * on_train_end(self) -- method that is executed when training starts
        (in the end of ```trainer.train```)

    * on_epoch_begin(self) -- method that is executed when the epoch starts
        (in the beginning of ```train_one_epoch```, ```validate_one_epoch```,
        ```predict```)

    * on_epoch_end(self) -- method that is executed when the epoch starts
        (in the end of ```train_one_epoch```, ```validate_one_epoch```,
        ```predict```)

    * on_batch_begin(self) -- method that is executed when the batch starts
        (in the beginning of training cycle body in ```train_one_epoch```,
        ```validate_one_epoch```, ```predict```)

    * on_batch_end(self) -- method that is executed when the batch starts
        (in the end of training cycle body in ```train_one_epoch```,
        ```validate_one_epoch```, ```predict```)

    * set_trainer(self, trainer) -- method that links the trainer to the callback.
    '''

    def __init__(self):
        pass

    def on_init(self):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_batch_begin(self):
        pass

    def on_batch_run(self):
        pass

    def on_batch_end(self):
        pass

    # def on_dataloader_ready(self):
    #     pass

    def __str__(self):
        pass

    def set_trainer(self, trainer):
        self.trainer = trainer

    def set_priority(self, priority):
        self.priority = priority