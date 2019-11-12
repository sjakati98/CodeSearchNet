from typing import Dict, Any, Union, Tuple

import tensorflow as tf
import tensorflow_hub as hub

from .seq_encoder import SeqEncoder
from utils.tfutils import write_to_feed_dict, pool_sequence_embedding


## Define constants for hyperparameters
## Embedding Type
ELMO = 'elmo'
LSTM1 = 'lstm1'
LSTM2 = 'lstm2'
WORD = 'word_emb'
## Pool Type
ELMO_FINAL = 'elmo_final'
ELMO_MEAN = 'elmo_weighted_mean'

class ElmoEncoder(SeqEncoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:

        ## TODO: set up the appropriate hyper parameters for the elmo model
        encoder_hypers = {
            'embedding_type': ELMO, ## One of [elmo, lstm1, lstm2, word_emb]
            'elmo_pool_mode': ELMO_MEAN
        }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)

    @property
    def output_representation_size(self):
        if self.get_hyper('embedding_type') is WORD:
            return 512
        else:
            return 1024
    
    def make_model(self, is_train: bool=False) -> tf.Tensor :
        with tf.variable_scope("elmo_encoder"):
            self._make_placeholders()

            self.placeholders['tokens_lengths'] = \
                tf.placeholder(tf.int32,
                               shape=[None],
                               name='tokens_lengths')

            seq_tokens = self.placeholders['tokens']
            seq_tokens_embeddings = self.embedding_layer(seq_tokens)
            seq_tokens_lengths = self.placeholders['tokens_lengths']


            ## pull elmo model from tensorflow hub
            elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=is_train)
            token_embeddings = elmo(
                {
                    "tokens": seq_tokens,
                    "squence_len": seq_tokens_lengths
                },
                signature='tokens',
                as_dict=True
            )[self.get_hyper('embedding_type')] ## [batch_size, max_length, 1024 or 512]

            output_pool_mode = self.get_hyper('elmo_pool_mode').lower()
            if output_pool_mode is ELMO_FINAL:
                return token_embeddings
            else:
                token_mask = tf.expand_dims(tf.range(tf.shape(seq_tokens)[1]), axis=0)            # 1 x T
                token_mask = tf.tile(token_mask, multiples=(tf.shape(seq_tokens_lengths)[0], 1))  # B x T
                token_mask = tf.cast(token_mask < tf.expand_dims(seq_tokens_lengths, axis=-1),
                                     dtype=tf.float32)                                            # B x T
                return pool_sequence_embedding(output_pool_mode,
                                               sequence_token_embeddings=token_embeddings,
                                               sequence_lengths=seq_tokens_lengths,
                                               sequence_token_masks=token_mask)

    def init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super().init_minibatch(batch_data)
        batch_data['tokens'] = []
        batch_data['tokens_lengths'] = []

    def minibatch_to_feed_dict(self, batch_data: Dict[str, Any], feed_dict: Dict[tf.Tensor, Any], is_train: bool) -> None:
        super().minibatch_to_feed_dict(batch_data, feed_dict, is_train)

        write_to_feed_dict(feed_dict, self.placeholders['tokens'], batch_data['tokens'])
        write_to_feed_dict(feed_dict, self.placeholders['tokens_lengths'], batch_data['tokens_lengths'])