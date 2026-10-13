"""Train a consistency model from random initialization on your labelled pairs."""
from train import main as train, run_cli


def main(cfg):
    return train(cfg,training_mode='cm')


if __name__=='__main__':
    run_cli('cm')
