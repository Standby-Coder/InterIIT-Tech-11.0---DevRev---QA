import tensorflow as tf
import tensorflow_datasets as tfds

# Load a QA dataset from TensorFlow Datasets
(train_dataset, val_dataset), dataset_info = tfds.load(
    'squad',
    split=['train[:80%]', 'train[80%:90%]'],
    with_info=True,
    download=True
)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(dataset_info.features['text'].encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(dataset_info.features['text'].encoder.vocab_size, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model on the QA dataset
model.fit(train_dataset, epochs=10)

# Use the trained model to answer questions
def answer_question(question, context):
    encoded_question = dataset_info.features['text'].encoder.encode([question])
    encoded_context = dataset_info.features['text'].encoder.encode([context])
    encoded_text = encoded_context + encoded_question
    logits = model(encoded_text)
    answer = dataset_info.features['text'].encoder.decode(tf.argmax(logits, axis=-1))
    return answer