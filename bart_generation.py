import torch
from preprocessor import Preprocessor
from config import init_argument_parser

if __name__ == "__main__":
    args = init_argument_parser().parse_args()
    test_dir = args.input_dir
    batch_size = args.batch_size
    output_path = args.output_path
    bart = torch.hub.load('pytorch/fairseq', 'bart.large.cnn')
    bart.eval()
    bart.half()
    if torch.cuda.is_available():
        bart = bart.cuda()
    count = 1
    tokenize = False

    cnn_preprocessor = Preprocessor(test_dir, tokenizer=None)
    content_list = cnn_preprocessor.get_document_summary(tokenize)
    summary_list = [" ".join(tup[1]) for tup in content_list]
    input_list = [" ".join(tup[0]) for tup in content_list]

    with open(output_path, 'w') as fout:
        for idx in range(0, len(input_list), batch_size):
            slines = input_list[idx: idx + batch_size]
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0,
                                               max_len_b=140, min_len=55, no_repeat_ngram_size=3)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []
