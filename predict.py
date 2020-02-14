import os
import sys

import numpy as np
import tensorflow as tf

from common import read_conll, process_sentences, load_ner_model
from common import encode, write_result
from common import argument_parser

import time
import datetime

# Workaround to use https://github.com/google-research/bert
# See https://github.com/google-research/bert/issues/977
tf.gfile = tf.io.gfile

def main(argv):
    argparser = argument_parser('predict')
    args = argparser.parse_args(argv[1:])

    if args.profile:
        log_dir="logs/"
        summary_writer = tf.summary.create_file_writer(log_dir + "predict/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    t0 = time.time()
    ner_model, tokenizer, labels, config = load_ner_model(args.ner_model_dir)
    t_load_ner_model = time.time() - t0
    max_seq_len = config['max_seq_length']

    label_map = { t: i for i, t in enumerate(labels) }
    inv_label_map = { v: k for k, v in label_map.items() }

    test_words, dummy_labels = read_conll(args.test_data, mode='test')
    test_data = process_sentences(test_words, dummy_labels, tokenizer,
                                  max_seq_len)

    test_x = encode(test_data.combined_tokens, tokenizer, max_seq_len)

    # warmup
    t0 = time.time()
    if not args.no_warmup:
        tmp = [test_x[0][0:1], test_x[1][0:1]]
        ner_model.predict(tmp, batch_size=args.batch_size, verbose=args.verbose)
    t_warmup = time.time() - t0

    n = len(test_x[0])
    if args.num_samples:
        n = args.num_samples
        test_x = [test_x[0][0:n], test_x[1][0:n]]

    print("Number of input samples: {}".format(len(test_x[0])))
    print("Batch size: {}".format(args.batch_size))

    if args.profile:
        tf.summary.trace_on(graph=True, profiler=True)

    t0 = time.time()
    probs = ner_model.predict(test_x, batch_size=args.batch_size, verbose=args.verbose)
    t_predict = time.time() - t0
    preds = np.argmax(probs, axis=-1)

    if args.profile:
        with summary_writer.as_default():
            tf.summary.trace_export(name="keras-bert-ner", step=0, profiler_outdir=log_dir)

    pred_labels = []
    for i, pred in enumerate(preds):
        pred_labels.append([inv_label_map[t] for t in 
                            pred[1:len(test_data.tokens[i])+1]])

    lines = write_result(
        args.output_file, test_data.words[0:n], test_data.lengths,
        test_data.tokens, test_data.labels, pred_labels, mode='predict'
    )

    print("load_ner_model: {:8.3f} sec".format(t_load_ner_model))
    print("warmup:         {:8.3f} sec".format(t_warmup))
    print("predict:        {:8.3f} sec {:8.3f} samples/sec".format(t_predict, len(test_x[0]) / t_predict))

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
