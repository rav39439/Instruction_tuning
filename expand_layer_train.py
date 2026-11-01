#-----------------------------------------------------------------------------------------------------

#EXPANDED LAYERS INSTRUCTION TUNING

# =========================================================
# INSTRUCTION TUNING FROM A PRETRAINED NEXT-TOKEN MODEL
# EXPAND 13 LAYERS TO 19 LAYERS
# TRAIN ONLY THE 6 NEW LAYERS
# =========================================================

import os
import json
import math
import numpy as np
import tensorflow as tf

from transformers import BertTokenizerFast


# =========================================================
# 0. CONFIGURATION
# =========================================================

DATA_FILE = (
    "/kaggle/input/datasets/ravckumar/combined-question-answers4/"
    "combined_questions_answers.json"
)

PRETRAINED_WEIGHTS = (
    "/kaggle/input/datasets/ravckumar/model427-upgrade/"
    "last_epoch_latest_transformer.weights.h5"
)

OUTPUT_DIR = "/kaggle/working/instruction_tuning"

SEQ_LEN = 256
TRAIN_RATIO = 0.97
SEED = 42

PER_GPU_BATCH_SIZE = 40
EPOCHS = 40
WARMUP_EPOCHS = 0.1

INITIAL_LR1 = 1e-5
MIN_LR1 = 1e-6
WEIGHT_DECAY = 0.01

# =========================================================
# TRANSFORMER EXPANSION CONFIGURATION
# =========================================================

BASE_NUM_LAYERS = 13
NEW_LAYERS = 6
TOTAL_NUM_LAYERS = BASE_NUM_LAYERS + NEW_LAYERS

# Use a fresh optimizer for the expanded architecture.
# Set True only when resuming from an optimizer state saved
# with this same 19-layer / six-trainable-layer configuration.

LOAD_OPTIMIZER_STATE = False

OPTIMIZER_STATE_PREFIX = os.path.join(
    OUTPUT_DIR,
    "optimizer_state"
)

# These must match your existing model configuration.
# DecoderOnlyTransformer must already be defined.
#
# EMBED_DIM = ...
# NUM_HEADS = ...
# DFF = ...
# DROPOUT = ...

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================================================
# 1. LOAD JSON QUESTION-ANSWER DATA
# =========================================================

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    raise ValueError(
        "Expected a JSON list of question-answer objects."
    )

qa_pairs = []

for item in data:

    if not isinstance(item, dict):
        continue

    question = str(item.get("question", "")).strip()
    answer = str(item.get("answer", "")).strip()

    if question and answer:
        qa_pairs.append((question, answer))

if len(qa_pairs) < 2:
    raise ValueError(
        "At least two valid question-answer pairs are required."
    )

print("Total valid QA pairs:", len(qa_pairs))


# =========================================================
# 2. LOAD THE PRETRAINED TOKENIZER
# =========================================================

PRETRAINED_TOKENIZER_DIR = (
    "/kaggle/input/datasets/ravckumar/instruction_tuning3/"
    "tokenizer"
)

if os.path.isdir(PRETRAINED_TOKENIZER_DIR):

    tokenizer = BertTokenizerFast.from_pretrained(
        PRETRAINED_TOKENIZER_DIR,
        local_files_only=True
    )

    print("Pretrained tokenizer loaded successfully.")

else:

    tokenizer = BertTokenizerFast.from_pretrained(
        "bert-base-uncased"
    )

    print("Pretrained tokenizer directory unavailable.")
    print("Loaded bert-base-uncased tokenizer instead.")

# Do not add new special tokens.
# The model must retain its original vocabulary.

vocab_size = tokenizer.vocab_size
PAD_ID = tokenizer.pad_token_id

print("Vocabulary size:", vocab_size)
print("Padding token ID:", PAD_ID)


# =========================================================
# 3. ENCODE INSTRUCTION-TUNING EXAMPLES
# =========================================================

# Text format:
#
# Question: What is Google App Engine?
# Answer: Google App Engine is a managed PaaS...
# End of answer.
#
# The ending phrase is ordinary text, not a special token.

QUESTION_PREFIX_IDS = tokenizer.encode(
    "Question: ",
    add_special_tokens=False
)

ANSWER_PREFIX_IDS = tokenizer.encode(
    "\nAnswer: ",
    add_special_tokens=False
)

END_IDS = tokenizer.encode(
    "\nEnd of answer.",
    add_special_tokens=False
)


def encode_qa(question, answer, seq_len):

    question_ids = tokenizer.encode(
        question,
        add_special_tokens=False
    )

    answer_ids = tokenizer.encode(
        answer,
        add_special_tokens=False
    )

    # Reserve space for prefixes, ending phrase,
    # and at least one answer token.

    fixed_overhead = (
        len(QUESTION_PREFIX_IDS)
        + len(ANSWER_PREFIX_IDS)
        + len(END_IDS)
    )

    max_question_tokens = max(
        0,
        seq_len - fixed_overhead - 1
    )

    question_ids = question_ids[:max_question_tokens]

    prefix_ids = (
        QUESTION_PREFIX_IDS
        + question_ids
        + ANSWER_PREFIX_IDS
    )

    # Leave space for the answer and ending phrase.

    answer_capacity = (
        seq_len - len(prefix_ids) - len(END_IDS)
    )

    answer_ids = answer_ids[:answer_capacity]

    if not answer_ids:
        return None

    # Complete sequence:
    # Question prefix + question + answer prefix
    # + answer + ending phrase.

    token_ids = (
        prefix_ids
        + answer_ids
        + END_IDS
    )

    # Next-token prediction.

    input_ids = token_ids[:-1]
    target_ids = token_ids[1:]

    # First answer token position in the target sequence.

    first_answer_target = len(prefix_ids) - 1

    token_weights = np.zeros(
        len(target_ids),
        dtype=np.float32
    )

    # Only answer tokens and the ending phrase
    # contribute to the training loss.

    token_weights[first_answer_target:] = 1.0

    # Pad to a fixed input length of seq_len - 1.

    padding_length = (
        (seq_len - 1) - len(input_ids)
    )

    input_ids += [PAD_ID] * padding_length
    target_ids += [PAD_ID] * padding_length

    token_weights = np.pad(
        token_weights,
        (0, padding_length)
    )

    return (
        input_ids,
        target_ids,
        token_weights
    )


# =========================================================
# 4. BUILD ARRAYS
# =========================================================

examples = []

for question, answer in qa_pairs:

    encoded = encode_qa(
        question,
        answer,
        SEQ_LEN
    )

    if encoded is not None:
        examples.append(encoded)

if len(examples) < 2:
    raise ValueError(
        "Not enough valid examples after tokenization."
    )

inputs = np.asarray(
    [example[0] for example in examples],
    dtype=np.int32
)

targets = np.asarray(
    [example[1] for example in examples],
    dtype=np.int32
)

sample_weights = np.asarray(
    [example[2] for example in examples],
    dtype=np.float32
)

print("Encoded examples:", len(examples))
print("Input shape:", inputs.shape)
print("Target shape:", targets.shape)
print("Weights shape:", sample_weights.shape)

print(
    "Average supervised tokens:",
    round(float(sample_weights.sum(axis=1).mean()), 2)
)


# =========================================================
# 5. TRAIN / VALIDATION SPLIT
# =========================================================

indices = np.random.permutation(len(inputs))

split_index = int(
    len(indices) * TRAIN_RATIO
)

# Ensure both datasets have at least one example.

split_index = min(
    max(split_index, 1),
    len(indices) - 1
)

train_indices = indices[:split_index]
val_indices = indices[split_index:]

train_inputs = inputs[train_indices]
train_targets = targets[train_indices]
train_weights = sample_weights[train_indices]

val_inputs = inputs[val_indices]
val_targets = targets[val_indices]
val_weights = sample_weights[val_indices]

print("Training examples:", len(train_inputs))
print("Validation examples:", len(val_inputs))


# =========================================================
# 6. DISTRIBUTED TRAINING DATASETS
# =========================================================

strategy = tf.distribute.MirroredStrategy()

NUM_GPUS = strategy.num_replicas_in_sync

GLOBAL_BATCH_SIZE = (
    PER_GPU_BATCH_SIZE * NUM_GPUS
)

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        train_inputs,
        train_targets,
        train_weights
    )
)

train_dataset = (
    train_dataset
    .shuffle(
        len(train_inputs),
        seed=SEED,
        reshuffle_each_iteration=True
    )
    .batch(
        GLOBAL_BATCH_SIZE,
        drop_remainder=True
    )
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (
        val_inputs,
        val_targets,
        val_weights
    )
)

val_dataset = (
    val_dataset
    .batch(
        GLOBAL_BATCH_SIZE,
        drop_remainder=False
    )
    .prefetch(tf.data.AUTOTUNE)
)

steps_per_epoch = len(train_dataset)
validation_steps = len(val_dataset)

if steps_per_epoch == 0:
    raise ValueError(
        "No training batches. Reduce the batch size or "
        "add more training examples."
    )
total_steps = steps_per_epoch * EPOCHS

warmup_steps = int(
    steps_per_epoch * WARMUP_EPOCHS
)

print("Number of GPUs:", NUM_GPUS)
print("Per-GPU batch size:", PER_GPU_BATCH_SIZE)
print("Global batch size:", GLOBAL_BATCH_SIZE)
print("Steps per epoch:", steps_per_epoch)
print("Validation steps:", validation_steps)
print("Total steps:", total_steps)
print("Warmup steps:", warmup_steps)


# =========================================================
# 7. WARMUP + COSINE LEARNING-RATE SCHEDULE
# =========================================================

class WarmupCosineDecay(
    tf.keras.optimizers.schedules.LearningRateSchedule
):

    def __init__(
        self,
        initial_lr,
        min_lr,
        warmup_steps,
        total_steps
    ):

        super().__init__()

        self.initial_lr = float(initial_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)

    def __call__(self, step):

        step = tf.cast(step, tf.float32)

        warmup_steps = tf.cast(
            self.warmup_steps,
            tf.float32
        )

        total_steps = tf.cast(
            self.total_steps,
            tf.float32
        )

        warmup_lr = (
            self.initial_lr
            * (step + 1.0)
            / tf.maximum(warmup_steps, 1.0)
        )

        progress = (
            (step - warmup_steps)
            / tf.maximum(
                total_steps - warmup_steps,
                1.0
            )
        )

        progress = tf.clip_by_value(
            progress,
            0.0,
            1.0
        )

        cosine_lr = (
            self.min_lr
            + 0.5
            * (self.initial_lr - self.min_lr)
            * (
                1.0
                + tf.cos(math.pi * progress)
            )
        )

        return tf.where(
            step < warmup_steps,
            warmup_lr,
            cosine_lr
        )

    def get_config(self):

        return {
            "initial_lr": self.initial_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps
        }


lr_schedule = WarmupCosineDecay(
    initial_lr=INITIAL_LR1,
    min_lr=MIN_LR1,
    warmup_steps=warmup_steps,
    total_steps=total_steps
)


# =========================================================
# 8. BUILD, LOAD AND EXPAND TRANSFORMER
# =========================================================

def expand_transformer_layers(
    model,
    num_new_layers,
    flag=True
):
    """
    Create an expanded DecoderOnlyTransformer.

    Original embedding, decoder layers and LM head are copied.
    Original layers, embedding and LM head are frozen.
    Only newly added decoder layers are trainable.

    New layers are initialized as identity-like residual blocks
    by zero-initializing their attention output and final FFN
    projections. This helps preserve the original model's logits
    immediately after expansion.
    """

    if not flag:
        print("Expansion disabled. Returning original model.")
        return model

    if (
        not isinstance(num_new_layers, int)
        or num_new_layers <= 0
    ):
        raise ValueError(
            "num_new_layers must be a positive integer."
        )

    if not model.built:
        raise ValueError(
            "Build the original model before expanding it."
        )

    original_layers = len(model.dec_layers)
    total_layers = original_layers + num_new_layers

    print("\n======================================")
    print("TRANSFORMER EXPANSION")
    print("======================================")
    print("Original layers:", original_layers)
    print("New layers:", num_new_layers)
    print("Total layers:", total_layers)

    expanded_model = DecoderOnlyTransformer(
        num_layers=total_layers,
        d_model=model.d_model,
        num_heads=model.num_heads,
        dff=model.dff,
        vocab_size=model.vocab_size,
        maximum_position_encoding=(
            model.maximum_position_encoding
        ),
        rate=model.rate
    )

    # Build all variables in the expanded architecture.

    dummy_input = tf.zeros(
        (1, model.maximum_position_encoding),
        dtype=tf.int32
    )

    expanded_model(
        dummy_input,
        training=False
    )

    # Copy embedding weights.

    expanded_model.embedding.set_weights(
        model.embedding.get_weights()
    )

    # Copy all original decoder-layer weights.

    for i in range(original_layers):

        expanded_model.dec_layers[i].set_weights(
            model.dec_layers[i].get_weights()
        )

    # Copy the original language-model head.

    expanded_model.final_layer.set_weights(
        model.final_layer.get_weights()
    )

    # Initialize newly added layers as identity-like
    # residual blocks.

    for i in range(original_layers, total_layers):

        layer = expanded_model.dec_layers[i]

        # Zero attention output projection.

        layer.mha.dense.kernel.assign(
            tf.zeros_like(layer.mha.dense.kernel)
        )

        layer.mha.dense.bias.assign(
            tf.zeros_like(layer.mha.dense.bias)
        )

        # Zero final FFN projection.

        final_ffn_dense = layer.ffn.layers[-1]

        final_ffn_dense.kernel.assign(
            tf.zeros_like(final_ffn_dense.kernel)
        )

        final_ffn_dense.bias.assign(
            tf.zeros_like(final_ffn_dense.bias)
        )

    # Freeze original components.

    expanded_model.trainable = True

    expanded_model.embedding.trainable = False

    for i in range(original_layers):
        expanded_model.dec_layers[i].trainable = False

    expanded_model.final_layer.trainable = False

    # Train only the newly added decoder layers.

    for i in range(original_layers, total_layers):
        expanded_model.dec_layers[i].trainable = True

    print("\nLayer trainability:")

    for i, layer in enumerate(expanded_model.dec_layers):

        status = (
            "TRAINABLE"
            if layer.trainable
            else "FROZEN"
        )

        print(f"Layer {i + 1:02d}: {status}")

    print("\nEmbedding: FROZEN")
    print("LM head: FROZEN")
    print("Expansion completed.")

    return expanded_model


with strategy.scope():

    # -----------------------------------------------------
    # CREATE ORIGINAL 13-LAYER TRANSFORMER
    # -----------------------------------------------------

    model = DecoderOnlyTransformer(
        num_layers=BASE_NUM_LAYERS,
        d_model=EMBED_DIM,
        num_heads=NUM_HEADS,
        dff=DFF,
        vocab_size=vocab_size,
        maximum_position_encoding=SEQ_LEN,
        rate=DROPOUT
    )

    # -----------------------------------------------------
    # BUILD ORIGINAL MODEL
    # -----------------------------------------------------

    dummy_input = tf.zeros(
        (1, SEQ_LEN - 1),
        dtype=tf.int32
    )

    dummy_output = model(
        dummy_input,
        training=False
    )

    print("\nOriginal model output shape:", dummy_output.shape)
    print("Original decoder layers:", len(model.dec_layers))
    print("Original total parameters:", model.count_params())

    # -----------------------------------------------------
    # LOAD ORIGINAL NEXT-TOKEN PRETRAINED WEIGHTS
    # -----------------------------------------------------

    if not os.path.exists(PRETRAINED_WEIGHTS):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {PRETRAINED_WEIGHTS}"
        )

    print("\nLoading original next-token pretrained weights...")

    model.load_weights(PRETRAINED_WEIGHTS)

    print("Original pretrained weights loaded successfully.")

    if len(model.dec_layers) != BASE_NUM_LAYERS:
        raise ValueError(
            f"Expected {BASE_NUM_LAYERS} layers in the original model, "
            f"but found {len(model.dec_layers)}."
        )

    original_parameters = model.count_params()

    # -----------------------------------------------------
    # EXPAND TRANSFORMER BY SIX NEW LAYERS
    # -----------------------------------------------------

    print("\nExpanding Transformer...")

    model = expand_transformer_layers(
        model=model,
        num_new_layers=NEW_LAYERS,
        flag=True
    )

    # -----------------------------------------------------
    # VERIFY EXPANDED MODEL
    # -----------------------------------------------------

    print("\n======================================")
    print("EXPANDED TRANSFORMER VERIFICATION")
    print("======================================")

    print("Original layers:", BASE_NUM_LAYERS)
    print("Added layers:", NEW_LAYERS)
    print("Total decoder layers:", len(model.dec_layers))
    print("Original parameters:", original_parameters)
    print("Expanded parameters:", model.count_params())

    if len(model.dec_layers) != TOTAL_NUM_LAYERS:
        raise ValueError(
            f"Expected {TOTAL_NUM_LAYERS} layers, "
            f"but found {len(model.dec_layers)}."
        )

    if model.count_params() <= original_parameters:
        raise ValueError(
            "Parameter count did not increase after expansion."
        )

    trainable_parameters = sum(
        int(tf.size(variable).numpy())
        for variable in model.trainable_variables
    )

    print("Trainable parameters:", trainable_parameters)

    print("\nLayer trainability:")

    for i, layer in enumerate(model.dec_layers):

        status = (
            "TRAINABLE"
            if layer.trainable
            else "FROZEN"
        )

        print(f"Layer {i + 1:02d}: {status}")

    print("\nEmbedding trainable:", model.embedding.trainable)
    print("LM head trainable:", model.final_layer.trainable)

    # -----------------------------------------------------
    # LOSS FUNCTION
    # -----------------------------------------------------

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    )

    # -----------------------------------------------------
    # CREATE FRESH OPTIMIZER
    # -----------------------------------------------------

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=WEIGHT_DECAY
    )

    # -----------------------------------------------------
    # COMPILE EXPANDED MODEL
    # -----------------------------------------------------

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        weighted_metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ]
    )

    # Build optimizer slots for the six newly trainable layers.

    if hasattr(optimizer, "build"):
        optimizer.build(model.trainable_variables)

    # Optional: restore optimizer state only when resuming from
    # a checkpoint created with this same expanded architecture
    # and the same trainable-variable configuration.

    if LOAD_OPTIMIZER_STATE:

        optimizer_index_path = (
            OPTIMIZER_STATE_PREFIX + ".index"
        )

        if not os.path.exists(optimizer_index_path):
            raise FileNotFoundError(
                "Optimizer state was requested, but no saved "
                "optimizer state was found at: "
                f"{OPTIMIZER_STATE_PREFIX}"
            )

        optimizer_status = tf.train.Checkpoint(
            optimizer=optimizer
        ).restore(OPTIMIZER_STATE_PREFIX)

        optimizer_status.assert_existing_objects_matched()

        print(
            "Optimizer state restored:",
            OPTIMIZER_STATE_PREFIX
        )

    print("\nExpanded model compiled successfully.")
    print("Only the six newly added layers are trainable.")


# ---------------------------------------------------------
# MODEL SUMMARY
# ---------------------------------------------------------

model.summary()


# =========================================================
# 9. CHECKPOINT CALLBACK
# =========================================================

checkpoint_path = os.path.join(
    OUTPUT_DIR,
    "instruction_tuned.weights.h5"
)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1
)


# =========================================================
# 10. INSTRUCTION TUNING
# =========================================================

print("\n====================================")
print("STARTING INSTRUCTION TUNING")
print("====================================")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb]
)


# =========================================================
# 11. EVALUATION
# =========================================================

print("\n====================================")
print("FINAL VALIDATION")
print("====================================")

results = model.evaluate(
    val_dataset,
    return_dict=True,
    verbose=1
)

for name, value in results.items():
    print(f"{name}: {value:.6f}")


# =========================================================
# 12. SAVE FINAL EPOCH WEIGHTS
# =========================================================

final_weights_path = os.path.join(
    OUTPUT_DIR,
    "final_instruction_tuned.weights.h5"
)

model.save_weights(final_weights_path)

print("Final weights saved:", final_weights_path)

# Save optimizer state separately.
# This stores optimizer iterations/slots for continued training.
# It is separate from the model weights file.

tf.train.Checkpoint(
    optimizer=optimizer
).write(OPTIMIZER_STATE_PREFIX)

print("Optimizer state saved:", OPTIMIZER_STATE_PREFIX)

# Save tokenizer with the model artifacts.

tokenizer.save_pretrained(
    os.path.join(OUTPUT_DIR, "tokenizer")
)

print("Tokenizer saved.")


# =========================================================
# 13. GENERATE ANSWERS
# =========================================================

def generate_answer(
    model,
    tokenizer,
    question,
    seq_len=SEQ_LEN,
    max_gen=150
):

    prompt = f"Question: {question}\nAnswer:"

    token_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False
    )

    if not token_ids:
        raise ValueError(
            "Question produced no tokens."
        )

    # Retain the most recent context.

    token_ids = token_ids[-(seq_len - 1):]

    input_seq = tf.constant(
        [token_ids],
        dtype=tf.int32
    )

    generated_ids = []

    for _ in range(max_gen):

        logits = model(
            input_seq,
            training=False
        )

        next_id = tf.argmax(
            logits[:, -1, :],
            axis=-1,
            output_type=tf.int32
        )

        next_id_value = int(next_id.numpy()[0])

        generated_ids.append(next_id_value)

        # Detect the ordinary-text end marker by its
        # encoded token sequence.

        if (
            len(generated_ids) >= len(END_IDS)
            and generated_ids[-len(END_IDS):] == END_IDS
        ):

            generated_ids = generated_ids[:-len(END_IDS)]
            break

        input_seq = tf.concat(
            [
                input_seq,
                tf.reshape(next_id, (1, 1))
            ],
            axis=1
        )

        if input_seq.shape[1] > seq_len - 1:
            input_seq = input_seq[:, -(seq_len - 1):]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()


# =========================================================
# 14. TEST THE INSTRUCTION-TUNED MODEL
# =========================================================

question = "What is Google App Engine?"

answer = generate_answer(
    model,
    tokenizer,
    question,
    max_gen=150
)

print("\nQuestion:", question)
print("Generated answer:", answer)
