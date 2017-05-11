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
        string = "{}\n{}\n{}\n{}\n{}\n".format(
            "Accuracy: {}".format(self.acc),
            "Value accuracy: {}".format(self.val_acc),
            "Loss: {}".format(self.loss),
            "Value loss: {}".format(self.val_loss),
            "Time to train: {}".format(self.time))
        return string
