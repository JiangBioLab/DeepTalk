import json
import pickle
import sys


def load_pickle(filename):
    try:
        with open(str(filename), "rb") as f:
            obj = pickle.load(f)

    except EOFError:
        obj = None

    return obj


def show_progress(curr_, total_, message=""):
    """Display progress."""
    prog_ = int(round(100.0 * float(curr_) / float(total_)))
    dstr = "[" + ">" * int(round(prog_ / 4)) + " " * (25 - int(round(prog_ / 4))) + "]"
    sys.stdout.write("{}{}% {}\r".format(dstr, prog_, message))
    sys.stdout.flush()


class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.lowest_loss = None

    def check_early_stopping(self, val_loss):
        if self.lowest_loss is None:
            self.lowest_loss = val_loss
            return False

        if val_loss + self.delta <= self.lowest_loss:
            self.counter += 1
            if self.counter >= self.patience:
                # stop model
                return True
        return False


def get_id_map(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data
