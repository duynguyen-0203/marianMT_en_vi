import argparse


def parse_args():
    r""""Parse all arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune MarianMT pretrained model for machine translation',
                                     allow_abbrev=False)

    parser = _add_data_args(parser)
    parser = _add_model_args(parser)
    parser = _add_train_args(parser)
    parser = _add_loss_args(parser)
    parser = _add_evaluation_args(parser)

    return parser


def _add_data_args(parser):
    parser.add_argument('--data_name', type=str, default='PhoMT', help='Name of the dataset')
    parser.add_argument('--src_train_data', type=str,
                        default=r'D:\Company\Machine_Translation\Data\PhoMT\tokenization\demo\demo.en',
                        help='Path to the file containing the source train data')
    parser.add_argument('--tgt_train_data', type=str,
                        default=r'D:\Company\Machine_Translation\Data\PhoMT\tokenization\demo\demo.vi',
                        help='Path to the file containing the target train data')
    parser.add_argument('--src_valid_data', type=str,
                        default=r'D:\Company\Machine_Translation\Data\PhoMT\tokenization\demo\demo.en',
                        help='Path to the file containing the source validation data')
    parser.add_argument('--tgt_valid_data', type=str,
                        default=r'D:\Company\Machine_Translation\Data\PhoMT\tokenization\demo\demo.vi',
                        help='Path to the file containing the target validation data')
    parser.add_argument('--max_length', type=int, default=64)
    return parser


def _add_model_args(parser):
    parser.add_argument('--model_name', type=str, default='marianMT')
    parser.add_argument('--model_path', type=str, default='Helsinki-NLP/opus-mt-en-vi',
                        help='The *model id* of a pretrained model configuration hosted inside a model repo on '
                             'huggingface.co')
    parser.add_argument('--src_lang', type=str, default='en_XX', help='Source language')
    parser.add_argument('--tgt_lang', type=str, default='vi_VN', help='Target language')

    return parser


def _add_train_args(parser):
    parser.add_argument('--save_path', type=str, default='training')
    parser.add_argument('--seed', type=int, default=36)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr_warmup', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--log_iter', type=int, default=200)

    return parser


def _add_loss_args(parser):

    return parser


def _add_evaluation_args(parser):
    parser.add_argument('--eval_batch_size', type=int, default=8)

    return parser
