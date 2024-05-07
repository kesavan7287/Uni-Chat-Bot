import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from insert import extract_text_from_pdf


def train_and_save_cbow_model(model_path, text = extract_text_from_pdf("F:\\psg\\sem_8\\nlp_package\\nlp_ca2_doc_package.pdf")):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1

    sequences = []
    for line in text.split('.'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(2, len(token_list) - 2):
            context = [token_list[i-2], token_list[i-1],
                       token_list[i+1], token_list[i+2]]
            target = token_list[i]
            sequences.append((context, target))

    X = []
    y = []
    for context, target in sequences:
        X.append(context)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    embedding_dim = 100
    cbow_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(total_words, embedding_dim, input_length=4),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(total_words, activation='softmax')
    ])

    cbow_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    cbow_model.fit(X, y, epochs=100, verbose=1)
    cbow_model.save(model_path)


def generate_word_embeddings(sentence, model_path):
    cbow_model = tf.keras.models.load_model(model_path)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([sentence])
    token_list = tokenizer.texts_to_sequences([sentence])[0]

    contexts = []
    for i in range(2, len(token_list) - 2):
        context = [token_list[i-2], token_list[i-1],
                   token_list[i+1], token_list[i+2]]
        contexts.append(context)

    embeddings = []
    for context in contexts:
        context = np.array(context).reshape(1, -1)
        embedding = cbow_model.layers[0](context)
        embedding = cbow_model.layers[1](embedding)
        embedding = cbow_model.layers[2](embedding)
        embedding = cbow_model.layers[3](embedding)
        embeddings.append(embedding.numpy())

    return np.mean(embeddings, axis=0).reshape(1, -1)


# train_and_save_cbow_model("F:\\psg\\sem_8\\nlp_package\\cbow_model_complex.h5")

# print(generate_word_embeddings("PSG College of Technology is located near Cbe", "F:\\psg\\sem_8\\nlp_package\\cbow_model_complex.h5"))