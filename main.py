import arguments

from src.trainer import Trainer


def _train(args):
    trainer = Trainer(args)
    trainer.train()


def _eval(args):
    trainer = Trainer(args)
    trainer.eval()


if __name__ == '__main__':
    parser = arguments.parse_args()
    args = parser.parse_args()

    if args.mode == 'train':
        _train(args)
    elif args.mode == 'eval':
        _eval(args)
