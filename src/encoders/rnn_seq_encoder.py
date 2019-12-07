from typing import Dict, Any, Union, Tuple

import tensorflow as tf

from .seq_encoder import SeqEncoder
from utils.tfutils import write_to_feed_dict, pool_sequence_embedding


def __make_rnn_cell(cell_type: str,
                    hidden_size: int,
                    dropout_keep_rate: Union[float, tf.Tensor]=1.0,
                    recurrent_dropout_keep_rate: Union[float, tf.Tensor]=1.0) \
        -> tf.nn.rnn_cell.RNNCell:
    """
    Args:
        cell_type: "lstm", "gru", or 'rnn' (any casing)
        hidden_size: size for the underlying recurrent unit
        dropout_keep_rate: output-vector dropout prob
        recurrent_dropout_keep_rate:  state-vector dropout prob

    Returns:
        RNNCell of the desired type.
    """
    cell_type = cell_type.lower()
    if cell_type == 'lstm':
        cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
    elif cell_type == 'gru':
        cell = tf.nn.rnn_cell.GRUCell(hidden_size)
    elif cell_type == 'rnn':
        cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
    else:
        raise ValueError("Unknown RNN cell type '%s'!" % cell_type)

    return tf.contrib.rnn.DropoutWrapper(cell,
                                         output_keep_prob=dropout_keep_rate,
                                         state_keep_prob=recurrent_dropout_keep_rate)


def _make_deep_rnn_cell(num_layers: int,
                        cell_type: str,
                        hidden_size: int,
                        dropout_keep_rate: Union[float, tf.Tensor]=1.0,
                        recurrent_dropout_keep_rate: Union[float, tf.Tensor]=1.0) \
        -> tf.nn.rnn_cell.RNNCell:
    """
    Args:
        num_layers: number of layers in result
        cell_type: "lstm" or "gru" (any casing)
        hidden_size: size for the underlying recurrent unit
        dropout_keep_rate: output-vector dropout prob
        recurrent_dropout_keep_rate: state-vector dropout prob

    Returns:
        (Multi)RNNCell of the desired type.
    """
    if num_layers == 1:
        return __make_rnn_cell(cell_type, hidden_size, dropout_keep_rate, recurrent_dropout_keep_rate)
    else:
        cells = [__make_rnn_cell(cell_type, hidden_size, dropout_keep_rate, recurrent_dropout_keep_rate)
                 for _ in range(num_layers)]
        return tf.nn.rnn_cell.MultiRNNCell(cells)


class RNNEncoder(SeqEncoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = {'rnn_num_layers': 2,
                          'rnn_hidden_dim': 64,
                          'rnn_cell_type': 'LSTM',  # One of [LSTM, GRU, RNN]
                          'rnn_is_bidirectional': True,
                          'rnn_dropout_keep_rate': 0.8,
                          'rnn_recurrent_dropout_keep_rate': 1.0,
                          'rnn_pool_mode': 'weighted_mean',
                          'rnn_do_attention': True,
                          }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)

    def _encode_with_rnn(self,
                         inputs: tf.Tensor,
                         input_lengths: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        cell_type = self.get_hyper('rnn_cell_type').lower()
        rnn_cell_fwd = _make_deep_rnn_cell(num_layers=self.get_hyper('rnn_num_layers'),
                                           cell_type=cell_type,
                                           hidden_size=self.get_hyper('rnn_hidden_dim'),
                                           dropout_keep_rate=self.placeholders['rnn_dropout_keep_rate'],
                                           recurrent_dropout_keep_rate=self.placeholders['rnn_recurrent_dropout_keep_rate'],
                                           )
        if not self.get_hyper('rnn_is_bidirectional'):
            (outputs, final_states) = tf.nn.dynamic_rnn(cell=rnn_cell_fwd,
                                                        inputs=inputs,
                                                        sequence_length=input_lengths,
                                                        dtype=tf.float32,
                                                        )

            if cell_type == 'lstm':
                final_state = tf.concat([tf.concat(layer_final_state, axis=-1)  # concat c & m of LSTM cell
                                         for layer_final_state in final_states],
                                        axis=-1)  # concat across layers
            elif cell_type == 'gru' or cell_type == 'rnn':
                final_state = tf.concat(final_states, axis=-1)
            else:
                raise ValueError("Unknown RNN cell type '%s'!" % cell_type)
        else:
            rnn_cell_bwd = _make_deep_rnn_cell(num_layers=self.get_hyper('rnn_num_layers'),
                                               cell_type=cell_type,
                                               hidden_size=self.get_hyper('rnn_hidden_dim'),
                                               dropout_keep_rate=self.placeholders['rnn_dropout_keep_rate'],
                                               recurrent_dropout_keep_rate=self.placeholders['rnn_recurrent_dropout_keep_rate'],
                                               )

            (outputs, final_states) = tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn_cell_fwd,
                                                                      cell_bw=rnn_cell_bwd,
                                                                      inputs=inputs,
                                                                      sequence_length=input_lengths,
                                                                      dtype=tf.float32,
                                                                      )
            # Merge fwd/bwd outputs:
            if cell_type == 'lstm':
                final_state = tf.concat([tf.concat([tf.concat(layer_final_state, axis=-1)  # concat c & m of LSTM cell
                                                    for layer_final_state in layer_final_states],
                                                   axis=-1)  # concat across layers
                                        for layer_final_states in final_states],
                                        axis=-1)  # concat fwd & bwd
            elif cell_type == 'gru' or cell_type == 'rnn':
                final_state = tf.concat([tf.concat(layer_final_states, axis=-1)  # concat across layers
                                         for layer_final_states in final_states],
                                        axis=-1)  # concat fwd & bwd
            else:
                raise ValueError("Unknown RNN cell type '%s'!" % cell_type)
            outputs = tf.concat(outputs, axis=-1)  # concat fwd & bwd

        return final_state, outputs

    def make_model(self, is_train: bool=False) -> tf.Tensor:
        with tf.variable_scope("rnn_encoder"):
            self._make_placeholders()

            self.placeholders['tokens_lengths'] = \
                tf.placeholder(tf.int32,
                               shape=[None],
                               name='tokens_lengths')

            self.placeholders['rnn_dropout_keep_rate'] = \
                tf.placeholder(tf.float32,
                               shape=[],
                               name='rnn_dropout_keep_rate')

            self.placeholders['rnn_recurrent_dropout_keep_rate'] = \
                tf.placeholder(tf.float32,
                               shape=[],
                               name='rnn_recurrent_dropout_keep_rate')

            self.seq_tokens = self.placeholders['tokens']
            seq_tokens_embeddings = self.embedding_layer(self.seq_tokens)
            seq_tokens_lengths = self.placeholders['tokens_lengths']

            rnn_final_state, self.token_embeddings = self._encode_with_rnn(seq_tokens_embeddings, seq_tokens_lengths)

            # TODO: Add call for Attention code.
            # Try to use batch queries so you can do bmm (TensorFlow equivalent)
            # Dim: batch_size, max_seq_len, emb_dim
            # Iterate over max_seq_len. For each token in sequence, do Attention
            #tf.map_fn -> runs a function over a set of values

            if (self.get_hyper('rnn_do_attention') == True):
                self.batch_seq_len = self.seq_tokens.get_shape().dims[1].value
                # self.attention = BahdanauAttention(self.batch_seq_len)
                # Do attention on each timestep
                batch_num = 100
                print("Starting Attention Setup")
                self.weights = tf.zeros([batch_num, 1, self.batch_seq_len])
                print("Set up Weights")
                self.ctx_v = tf.zeros(tf.shape(self.token_embeddings[:, 0:1, :]))
                print("Set up Context Vector")

                # run attention_hw_style on all tokens
                print("Running Attention")
                ctx_vec, attn_weights = tf.map_fn(self.attention_hw_style, tf.range(0, self.batch_seq_len, 1))

                print("Concatenating Context Vectors with Token Embeddings")
                # Concat context vectors and token_embeddings
                self.token_embeddings = tf.concat((self.ctx_v, self.token_embeddings), 1)

                print("Running the rest of the model")

            output_pool_mode = self.get_hyper('rnn_pool_mode').lower()
            if output_pool_mode == 'rnn_final':
                return rnn_final_state
            else:
                token_mask = tf.expand_dims(tf.range(tf.shape(self.seq_tokens)[1]), axis=0)       # 1 x T
                token_mask = tf.tile(token_mask, multiples=(tf.shape(seq_tokens_lengths)[0], 1))  # B x T
                token_mask = tf.cast(token_mask < tf.expand_dims(seq_tokens_lengths, axis=-1),
                                     dtype=tf.float32)                                            # B x T
                return pool_sequence_embedding(output_pool_mode,
                                               sequence_token_embeddings=self.token_embeddings,
                                               sequence_lengths=seq_tokens_lengths,
                                               sequence_token_masks=token_mask)

    '''
    # Code from TensorFlow
    def attention_helper(self, t):
        x = self.token_embeddings
        curr_hidden = x[:, t:t+1, :]
        prev_hiddens = x[:, :t, :]

        ctx_vec, attn_weights = self.attention(curr_hidden, prev_hiddens)

        return ctx_vec, attn_weights
    '''

    # Code from HW
    def attention_hw_style(self, t):
        x = self.token_embeddings
        curr_hidden = x[:, t:t+1, :]
        prev_hiddens = x[:, :t, :]

        prev_hiddens = tf.transpose(prev_hiddens, perm=[0, 2, 1])
        attn_score = tf.matmul(curr_hidden, prev_hiddens)
        attn_weight = tf.nn.softmax(attn_score, dim=2)

        attn_weight = tf.transpose(attn_weight, perm=[0, 2, 1])
        new_ctx = tf.matmul(prev_hiddens, attn_weight)

        # Concat Stuff
        new_ctx = tf.transpose(new_ctx, perm=[0, 2, 1])
        self.ctx_v = tf.concat((self.ctx_v, new_ctx), 1)
        attn_weight = tf.transpose(attn_weight, perm=[0, 2, 1])

        padding = tf.constant([[1, 0],[2, self.batch_seq_len-t]])
        attn_weight = tf.pad(attn_weight, paddings=padding, mode="CONSTANT", constant_values=0)
        self.weights = tf.concat((self.weights, attn_weight), 1)
        return new_ctx, attn_weight


    def init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super().init_minibatch(batch_data)
        batch_data['tokens'] = []
        batch_data['tokens_lengths'] = []

    def minibatch_to_feed_dict(self, batch_data: Dict[str, Any], feed_dict: Dict[tf.Tensor, Any], is_train: bool) -> None:
        super().minibatch_to_feed_dict(batch_data, feed_dict, is_train)
        feed_dict[self.placeholders['rnn_dropout_keep_rate']] = \
            self.get_hyper('rnn_dropout_keep_rate') if is_train else 1.0
        feed_dict[self.placeholders['rnn_recurrent_dropout_keep_rate']] = \
            self.get_hyper('rnn_recurrent_dropout_keep_rate') if is_train else 1.0

        write_to_feed_dict(feed_dict, self.placeholders['tokens'], batch_data['tokens'])
        write_to_feed_dict(feed_dict, self.placeholders['tokens_lengths'], batch_data['tokens_lengths'])
