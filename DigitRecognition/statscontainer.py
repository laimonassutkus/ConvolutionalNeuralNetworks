class Stats:
    acc = -1
    val_acc = -1
    loss = -1
    val_loss = -1
    time = -1

    def __init__(self, acc, val_acc, loss, val_loss, time):
        self.acc = acc
        self.val_acc = val_acc
        self.loss = loss
        self.val_loss = val_loss
        self.time = time

    def __str__(self):
        string = "{}\n{}\n{}\n{}\n{}".format(
            "Accuracy: {}".format(self.acc),
            "Value accuracy: {}".format(self.val_acc),
            "Loss: {}".format(self.loss),
            "Value loss: {}".format(self.val_loss),
            "Time to train: {}".format(self.time))
        return string


class GuessStats:
    false_count = -1
    correct_count = -1

    def __init__(self, falses=0, corrects=0):
        self.false_count = falses
        self.correct_count = corrects

    def increment_false(self):
        self.false_count += 1

    def increment_correct(self):
        self.correct_count += 1



