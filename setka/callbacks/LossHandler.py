from .Callback import Callback

class LossHandler(Callback):
    def __init__(self, criterion):
        self.criterion = criterion

        self.set_priority({'on_batch_run': 9, 'on_batch_end': -9})

    def on_batch_run(self):
        if self.trainer.status["mode"] in ["train", "valid"]:
            self.trainer._loss = self.criterion(self.trainer._output, self.trainer._input)
            if self.trainer.status["mode"] == "train":
                self.trainer._loss.backward()

            self.trainer.status['loss'] = self.trainer._loss.detach().cpu().item()

    def on_batch_end(self):
        if self.trainer.status["mode"] in ["train", "valid"]:
            del self.trainer._loss