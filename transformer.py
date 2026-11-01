## import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import string
import tensorflow as tf
import re

import os
import time
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import re
# -------------------------
# Parameters
# -------------------------

TRAIN_RATIO = 0.99
SEQ_LEN = 256
STRIDE = 256
# PER_GPU_BATCH_SIZE = 38
# GLOBAL_BATCH_SIZE  = 76
BATCH_SIZE = 24
BUFFER_SIZE = 10000
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 13
DFF = 1024
DROPOUT = 0.1
# T
EPOCHS =30
INITIAL_LR = 3e-5
MIN_LR = 8e-6
WARMUP_EPOCHS = 1
WEIGHT_DECAY = 0.01
# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
SEED = 42

# -------------------------
# 1. Read text file
# -------------------------



# --------------------------transformer---------------------------------------

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...].astype(np.float32)
    return tf.constant(pos_encoding)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

# -------------------------
# MHA and FFN (same as you had)
# -------------------------
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        # mask shape should broadcast to (..., seq_len_q, seq_len_k)
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

    
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q); k = self.wk(k); v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        out = self.dense(concat)
        return out, attn_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation=tf.nn.gelu),
        tf.keras.layers.Dense(d_model)
    ])

class DecoderOnlyLayer(tf.keras.layers.Layer):
    """Single decoder-only layer: masked self-attention + feedforward"""
    def __init__(self, d_model, num_heads, dff, rate=0.05):
        super(DecoderOnlyLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # Pre-LayerNorm
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, look_ahead_mask=None):
        # Pre-norm before masked self-attention
        x_norm = self.layernorm1(x)
        attn_output, _ = self.mha(x_norm, x_norm, x_norm, look_ahead_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = x + attn_output

        # Pre-norm before FFN
        out1_norm = self.layernorm2(out1)
        ffn_output = self.ffn(out1_norm)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = out1 + ffn_output

        return out2



class DecoderOnlyTransformer(tf.keras.Model):

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        vocab_size,
        maximum_position_encoding,
        rate=0.05,
        **kwargs
    ):
        super(DecoderOnlyTransformer, self).__init__(**kwargs)

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.vocab_size = vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            d_model
        )

        self.pos_encoding = positional_encoding(
            maximum_position_encoding,
            d_model
        )

        self.dec_layers = [
            DecoderOnlyLayer(
                d_model,
                num_heads,
                dff,
                rate
            )
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, x, training=False, look_ahead_mask=None):

        seq_len = tf.shape(x)[1]

        if look_ahead_mask is None:
            look_ahead_mask = create_look_ahead_mask(seq_len)

        x = self.embedding(x)

        x *= tf.math.sqrt(
            tf.cast(self.d_model, tf.float32)
        )

        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(
            x,
            training=training
        )

        for i in range(self.num_layers):

            x = self.dec_layers[i](
                x,
                training=training,
                look_ahead_mask=look_ahead_mask
            )

        logits = self.final_layer(x)

        return logits

    def get_config(self):

        config = super().get_config()

        config.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "vocab_size": self.vocab_size,
            "maximum_position_encoding": self.maximum_position_encoding,
            "rate": self.rate,
        })

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



def expand_transformer_layers(model, num_new_layers, flag=True):
    """
    Expand an existing DecoderOnlyTransformer without modifying
    its pretrained weights.

    Parameters
    ----------
    model : DecoderOnlyTransformer
        Existing, built Transformer model.

    num_new_layers : int
        Number of new Transformer layers to add.

    flag : bool
        True  -> Expand model and freeze original layers.
        False -> Return original model unchanged.

    Returns
    -------
    expanded_model : DecoderOnlyTransformer
    """

    print("\n" + "=" * 60)
    print("       TRANSFORMER ARCHITECTURE EXPANSION")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Check flag
    # --------------------------------------------------

    if not flag:

        print("Expansion disabled.")
        print("Returning original Transformer unchanged.")

        return model

    if not isinstance(num_new_layers, int) or num_new_layers <= 0:
        raise ValueError("num_new_layers must be a positive integer.")

    if not model.built:
        raise ValueError(
            "Build the original model before expanding it."
        )

    # --------------------------------------------------
    # 2. Record original architecture
    # --------------------------------------------------

    original_layers = len(model.dec_layers)

    total_layers = original_layers + num_new_layers

    print(f"\nExisting layers       : {original_layers}")
    print(f"New layers requested  : {num_new_layers}")
    print(f"Updated model depth   : {total_layers}")

    # --------------------------------------------------
    # 3. Create expanded Transformer
    # --------------------------------------------------

    expanded_model = DecoderOnlyTransformer(

        num_layers=total_layers,

        d_model=model.d_model,

        num_heads=model.num_heads,

        dff=model.dff,

        vocab_size=model.vocab_size,

        maximum_position_encoding=model.maximum_position_encoding,

        rate=model.rate

    )

    # --------------------------------------------------
    # 4. Build expanded model
    # --------------------------------------------------

    dummy_input = tf.zeros(
        (1, model.maximum_position_encoding),
        dtype=tf.int32
    )

    expanded_model(
        dummy_input,
        training=False
    )

    # --------------------------------------------------
    # 5. Copy original embedding weights
    # --------------------------------------------------

    expanded_model.embedding.set_weights(
        model.embedding.get_weights()
    )

    # --------------------------------------------------
    # 6. Copy original 13 Transformer layers
    # --------------------------------------------------

    for i in range(original_layers):

        expanded_model.dec_layers[i].set_weights(
            model.dec_layers[i].get_weights()
        )

    # --------------------------------------------------
    # 7. Copy original LM head
    # --------------------------------------------------

    expanded_model.final_layer.set_weights(
        model.final_layer.get_weights()
    )

    # --------------------------------------------------
    # 8. Initialize new layers as identity transformations
    # --------------------------------------------------
    # This is important:
    #
    # Each new layer initially produces approximately
    # its input unchanged.
    #
    # Therefore, adding new layers does not initially
    # disrupt the original model's predictions.
    #
    # The output projections will learn during training.

    for i in range(original_layers, total_layers):

        layer = expanded_model.dec_layers[i]

        # Attention output projection
        layer.mha.dense.kernel.assign(
            tf.zeros_like(layer.mha.dense.kernel)
        )

        layer.mha.dense.bias.assign(
            tf.zeros_like(layer.mha.dense.bias)
        )

        # Final FFN projection
        final_ffn_dense = layer.ffn.layers[-1]

        final_ffn_dense.kernel.assign(
            tf.zeros_like(final_ffn_dense.kernel)
        )

        final_ffn_dense.bias.assign(
            tf.zeros_like(final_ffn_dense.bias)
        )

    # --------------------------------------------------
    # 9. Freeze original architecture
    # --------------------------------------------------

    expanded_model.trainable = True

    # Freeze original embeddings
    expanded_model.embedding.trainable = False

    # Freeze original 13 Transformer layers
    for i in range(original_layers):

        expanded_model.dec_layers[i].trainable = False

    # Freeze original output projection
    expanded_model.final_layer.trainable = False

    # Keep newly added layers trainable
    for i in range(original_layers, total_layers):

        expanded_model.dec_layers[i].trainable = True

    # --------------------------------------------------
    # 10. Verify architecture
    # --------------------------------------------------

    print("\nOriginal weights transferred successfully.")

    print("\nLayer configuration:")

    for i, layer in enumerate(expanded_model.dec_layers):

        status = (
            "TRAINABLE"
            if layer.trainable
            else "FROZEN"
        )

        print(f"Layer {i + 1:02d} : {status}")

    print("\nEmbedding       : FROZEN")
    print("LM Head         : FROZEN")

    print("\nOriginal layers : ", original_layers)
    print("Added layers    : ", num_new_layers)
    print("Total layers    : ", total_layers)

    print(
        "\nTrainable variables:",
        len(expanded_model.trainable_variables)
    )

    print(
        "Trainable parameters:",
        sum(
            int(tf.size(v).numpy())
            for v in expanded_model.trainable_variables
        )
    )

    print("\n" + "=" * 60)
    print("       EXPANSION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return expanded_model

#-----------------------------------------------------------------------------------

