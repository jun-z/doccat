from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import time
import tensorflow as tf
from lstm import LSTM

tf.app.flags.DEFINE_string('data_dir', './data', 'Data directory.')
tf.app.flags.DEFINE_string('train_dir', './model', 'Training directory.')
tf.app.flags.DEFINE_integer('num_epochs', 5, 'Number of epochs.')
tf.app.flags.DEFINE_integer('batch_size', 20, 'Batch size.')
tf.app.flags.DEFINE_integer('num_units', 50, 'Number of units in LSTM.')
tf.app.flags.DEFINE_integer('num_layers', 1, 'Number of LSTM layers.')
tf.app.flags.DEFINE_integer('num_steps', 25, 'Max number of time steps')
tf.app.flags.DEFINE_integer('num_labels', 8, 'Number of labels.')
tf.app.flags.DEFINE_integer('emb_size', 10, 'Size of embedding.')
tf.app.flags.DEFINE_integer('vocab_size', 55, 'Size of vocabulary.')
tf.app.flags.DEFINE_float('learning_rate', .03, 'Learning rate.')
tf.app.flags.DEFINE_float('max_clip_norm', 5.0, 'Clip norm for gradients.')
tf.app.flags.DEFINE_bool('use_fp16', False, 'Use tf.float16.')
tf.app.flags.DEFINE_bool('do_label', False, 'Train model or label sequence.')

FLAGS = tf.app.flags.FLAGS


def create_model(session, batch):
    model = LSTM(
        batch,
        FLAGS.num_units,
        FLAGS.num_layers,
        FLAGS.num_steps,
        FLAGS.num_labels,
        FLAGS.emb_size,
        FLAGS.vocab_size,
        FLAGS.learning_rate,
        FLAGS.max_clip_norm,
        tf.float16 if FLAGS.use_fp16 else tf.float32)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Restoring model from %s.' % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print('Created model with fresh parameters.')
        session.run(tf.local_variables_initializer())
        session.run(tf.global_variables_initializer())
    return model


def train():
    train_files = []
    for f in os.listdir(FLAGS.data_dir):
        if f.endswith('.train.tfr'):
            train_files.append(os.path.join(FLAGS.data_dir, f))

    fn_queue = tf.train.string_input_producer(
        train_files, num_epochs=FLAGS.num_epochs)

    reader = tf.TFRecordReader()
    _, serialized = reader.read(fn_queue)

    context_features = {
        'label': tf.FixedLenFeature([], dtype=tf.int64),
        'length': tf.FixedLenFeature([], dtype=tf.int64),
        'weight': tf.FixedLenFeature([], dtype=tf.int64),
    }
    sequence_features = {
        'tokens': tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    context, sequence = tf.parse_single_sequence_example(
        serialized,
        context_features=context_features,
        sequence_features=sequence_features)

    example = {}
    example.update(context)
    example.update(sequence)

    batch = tf.train.batch(
        example, FLAGS.batch_size,
        allow_smaller_final_batch=True,
        shapes=[(), (), (FLAGS.num_steps), ()])

    with tf.Session() as sess:
        model = create_model(sess, batch)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                start = time.time()
                _, loss = sess.run([model.train, model.loss])
                duration = time.time() - start

                if step % 100 == 0:
                    print('Step %d: loss = %.2f (%.3f sec)' %
                          (step, loss, duration))
                    model.saver.save(
                        sess,
                        os.path.join(FLAGS.train_dir, 'doccat.ckpt'),
                        global_step=step)
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' %
                  (FLAGS.num_epochs, step))
        finally:
            coord.request_stop()

        coord.join(threads)


if __name__ == '__main__':
    if FLAGS.do_label:
        pass
    else:
        train()
