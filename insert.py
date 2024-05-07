from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from dotenv import load_dotenv


from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import re

load_dotenv(dotenv_path=os.path.join('.env'))


def extract_text_from_pdf(file_path="F:\\psg\\sem_8\\nlp_package\\nlp_ca2_doc_package_v2.pdf"):
    resource_manager = PDFResourceManager()
    return_list = []

    layout_params = LAParams()
    device = TextConverter(resource_manager, StringIO(), laparams=layout_params)
    interpreter = PDFPageInterpreter(resource_manager, device)

    with open(file_path, 'rb') as fp:
        password = ""
        max_pages = 0
        caching = True
        page_nos = set()
        for page in PDFPage.get_pages(fp, page_nos, maxpages=max_pages, password=password, caching=caching,
                                        check_extractable=True):
            return_str = StringIO()
            device = TextConverter(resource_manager, return_str, laparams=layout_params)
            interpreter = PDFPageInterpreter(resource_manager, device)
            interpreter.process_page(page)
            text = return_str.getvalue().lower()
            return_list.append(text)
            device.close()
            return_str.close()

    return return_list


def remove_punctuation(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    return cleaned_text


def remove_specific_words(text):
    words_to_remove = [
                          '\n', '|', '\t',
                      ]
    pattern = r'\b(?:{})\b'.format('|'.join(words_to_remove))
    cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return cleaned_text


def remove_stop_words(text):
    import string
    stop_words = set(stopwords.words("english"))
    punctuation = list(string.punctuation)
    stop_words.update(punctuation)

    text = " ".join(x for x in text.split() if x not in stop_words)
    return text


def remove_personal_info(text):
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove phone numbers
    text = re.sub(r'\d{3}[-\.\s]?\d{3}[-\.\s]?\d{4}', '', text)
    # Remove addresses
    text = re.sub(r'\d+\s+[\w\s]+,\s+[\w\s]+,\s+[\w\s]+', '', text)
    return text


def remove_numbers(text):
    text = re.sub(r'\d','',text)
    return text


def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


def preprocess_encode_text(text):
    text = remove_specific_words(text)
    text = remove_stop_words(text)
    text = remove_punctuation(text)
    text = lemmatize_text(text)



def generate_word_embeddings(sentence, model_path="F:\\psg\\sem_8\\nlp_package\\cbow_model_complex.h5"):
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


def insert(text, collection_name_qdrant):
    from qdrant_client import models, QdrantClient
    from qdrant_client.http.models import Batch
    text = generate_word_embeddings(extract_text_from_pdf())

    for i in text:
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

        # embeddings_response = cohere_client.embed(
        #     texts=[text],
        #     model="embed-english-v3.0",
        #     input_type="search_document"
        # )
        embeddings_response = generate_word_embeddings(i)
        

        vectors = [
            # Conversion to float is required for Qdrant
            list(map(float, vector)) 
            for vector in embeddings_response
        ]

        document = [
                {
                    "source": "local"
                }
            ]



        # Filling up Qdrant collection with the embeddings generated by Cohere co.embed API
        qdrant_client.upsert(
            collection_name="nlp_trials_v1", 
            points=Batch(
                ids=[counter],
                vectors=vectors,
                payloads=document,
            )
        )

        counter += 1


def insert_v2(collection_name_qdrant, file_path="F:\\psg\\sem_8\\nlp_package\\nlp_ca2_doc_package_v2.pdf"):
    from langchain.vectorstores import Qdrant
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.embeddings.cohere import CohereEmbeddings

    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
    embeddings = CohereEmbeddings(model = "embed-english-v3.0")

    qdrant = Qdrant.from_documents(
        pages,
        embeddings,
        url=os.getenv("QDRANT_URL"),
        prefer_grpc=True,
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=collection_name_qdrant,
    )