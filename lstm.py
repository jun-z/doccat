import tensorflow as tf


class LSTM(object):
    def __init__(self,
                 batch,
                 num_units,
                 num_layers,
                 num_steps,
                 num_labels,
                 emb_size,
                 vocab_size,
                 learning_rate,
                 max_clip_norm,
                 dtype=tf.float32):

        self.label = batch['label']
        self.tokens = batch['tokens']
        self.length = batch['length']
        self.weight = tf.cast(batch['weight'], dtype)

        embedding = tf.get_variable(
            'embedding', [vocab_size, emb_size], dtype=dtype)

        inp_emb = tf.nn.embedding_lookup(embedding, self.tokens)

        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

        outputs, _ = tf.nn.rnn(
            cell=cell,
            inputs=tf.unstack(inp_emb, axis=1),
            dtype=dtype,
            sequence_length=self.length)

        W = tf.get_variable(
            'W', [num_units, num_labels], dtype=dtype,
            initializer=tf.truncated_normal_initializer(stddev=.01))
        b = tf.get_variable(
            'b', [num_labels], dtype=dtype,
            initializer=tf.constant_initializer(.1))

        logits = tf.matmul(outputs[-1], W) + b
        targets = tf.one_hot(self.label, num_labels, dtype=dtype)

        xentropy = tf.nn.softmax_cross_entropy_with_logits(logits, targets)
        weighted = tf.mul(xentropy, self.weight)

        self.loss = tf.reduce_sum(weighted) / tf.reduce_sum(self.weight)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        params = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, params), max_clip_norm)

        self.probs = tf.nn.softmax(logits)
        self.train = optimizer.apply_gradients(zip(grads, params))
        self.saver = tf.train.Saver(tf.global_variables())
