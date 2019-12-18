class Pipe:
    '''
    pipe basic class.

    pipe has the following methods:

    * __init__(self) -- constructor

    * on_train_begin(self) -- method that is executed when training starts
        (in the beginning of ```trainer.train```)

    * on_train_end(self) -- method that is executed when training starts
        (in the end of ```trainer.train```)

    * before_epoch(self) -- method that is executed when the epoch starts
        (in the beginning of ```train_one_epoch```, ```validate_one_epoch```,
        ```predict```)

    * after_epoch(self) -- method that is executed when the epoch starts
        (in the end of ```train_one_epoch```, ```validate_one_epoch```,
        ```predict```)

    * before_batch(self) -- method that is executed when the batch starts
        (in the beginning of training cycle body in ```train_one_epoch```,
        ```validate_one_epoch```, ```predict```)

    * after_batch(self) -- method that is executed when the batch starts
        (in the end of training cycle body in ```train_one_epoch```,
        ```validate_one_epoch```, ```predict```)

    * set_trainer(self, trainer) -- method that links the trainer to the pipe.
    '''

    def __init__(self):
        self.trainer = None

    def on_init(self):
        pass

    def set_priority(self, priority):
        self.priority = priority