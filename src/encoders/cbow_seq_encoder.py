from typing import Dict, Any

import tensorflow as tf

from .masked_seq_encoder import MaskedSeqEncoder
from utils.tfutils import pool_sequence_embedding


class CBoWEncoder(MaskedSeqEncoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = { 'cbow_pool_mode': 'weighted_mean',
                         }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)

    def make_model(self, is_train: bool=False) -> tf.Tensor:
        with tf.variable_scope("cbow_encoder"):
            self._make_placeholders()

            self.seq_tokens_embeddings = self.embedding_layer(self.placeholders['tokens']) # batch size x max seq len x emb dim
            seq_token_mask = self.placeholders['tokens_mask']
            seq_token_lengths = tf.reduce_sum(seq_token_mask, axis=1)  # B

            batch_seq_len = self.seq_tokens_embeddings.get_shape().dims[1].value

            # pad seqs 
            paddings = tf.constant([[0, 0], [2, 2], [0, 0]])
            self.seq_tokens_embeddings = tf.pad(self.seq_tokens_embeddings, paddings, "CONSTANT")

            self.seq_tokens_embeddings = tf.map_fn(self.token_sums, tf.range(0, batch_seq_len, 1), parallel_iterations=1, dtype=(tf.float32)) # max seq len x batch size x emb dim

            # perm dims
            self.seq_tokens_embeddings = tf.transpose(self.seq_tokens_embeddings, perm=[1, 0, 2]) # batch size x max seq len x emb dim

            return pool_sequence_embedding(self.get_hyper('cbow_pool_mode').lower(),
                                           sequence_token_embeddings=self.seq_tokens_embeddings,
                                           sequence_lengths=seq_token_lengths,
                                           sequence_token_masks=seq_token_mask,
                                           is_train=is_train)

    def token_sums(self, t):
        x = self.seq_tokens_embeddings

        context = tf.concat( [ x[:, t:t+2, :], x[:, t+3:t+5, :] ], axis=1) # batch size x 4 x emb dim

        return tf.reduce_sum(context, 1) # batch size x emb dim
