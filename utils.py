from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


def serialize(label, tokens, length, weight):
    seq = tf.train.SequenceExample()

    seq.context.feature['label'].int64_list.value.append(label)
    seq.context.feature['length'].int64_list.value.append(length)
    seq.context.feature['weight'].int64_list.value.append(weight)

    _tokens = seq.feature_lists.feature_list['tokens']
    for t in tokens:
        _tokens.feature.add().int64_list.value.append(t)
    return seq.SerializeToString()


def write_records(path, labels, inputs, lengths, weights):
    writer = tf.python_io.TFRecordWriter(path)
    for label, tokens, length, weight in zip(labels, inputs, lengths, weights):
        writer.write(serialize(label, tokens, length, weight))
    writer.close()
