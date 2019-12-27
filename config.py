import argparse


def init_argument_parser():
    parser = argparse.ArgumentParser(description="Sentence Evaluation")

    parser.add_argument("--input-dir", type=str, metavar="N",
                        default="/home/zxj/Downloads/cnn/stories_test",
                        help="path of data directory")

    parser.add_argument("--batch-size", type=int,
                        default=24,
                        help="path of glove file")

    parser.add_argument("--use-cuda", action='store_true',
                        default=False,
                        help="whether to use cuda")

    parser.add_argument("--half-precision", action='store_true',
                        default=False,
                        help="whether to use half precision inference")

    parser.add_argument("--output-dir", type=str,
                        default="/home/zxj/Downloads/cnn/output")

    parser.add_argument("--use-multiple-gpu", action='store_true',
                        default=False, help='whether to use multiple gpus at inference stage')
    
    parser.add_argument("--model-name", type=str, default="gpt2", help="name of the pre-trained-model")

    parser.add_argument("--method", type=str, default="extractive", help="methods used to generate summarization")
    return parser
