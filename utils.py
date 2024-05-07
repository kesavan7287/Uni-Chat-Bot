from dotenv import load_dotenv
import os
from qdrant_client import models, QdrantClient
from gensim.models import Word2Vec
import numpy as np
from datetime import datetime
import pytz
import pymongo
from pymongo import MongoClient


load_dotenv(dotenv_path=os.path.join('.env'))


def create_collection(collection_name):
    client = QdrantClient(
        url = os.getenv("QDRANT_URL"),
        api_key = os.getenv("QDRANT_API_KEY"),
    )

    client.create_collection(
        collection_name = f"{collection_name}",
        vectors_config = models.VectorParams(size=1024, distance=models.Distance.COSINE),
    )


def delete_collection(collection_name):
    client = QdrantClient(
        url = os.getenv("QDRANT_URL"),
        api_key = os.getenv("QDRANT_API_KEY"),
    )

    client.delete_collection(collection_name = f"{collection_name}")


def get_collection(db_name = "nlp", collection_name = "nlp_package_v1"):
    client = MongoClient(os.getenv("MONGO_DB_URL"))
    db = client[db_name]
    collection = db[collection_name]

    return collection


def insert_data(user_id, session_id, query, response, collection = get_collection()):
    # Get the current UTC time
    current_time_utc = datetime.utcnow()

    # Define the IST timezone
    ist_timezone = pytz.timezone('Asia/Kolkata')

    # Convert the UTC time to IST
    current_time_ist = current_time_utc.astimezone(ist_timezone)

    # Create a document to insert
    data_to_insert = {
        # "_id": ObjectId(),  # Use ObjectId to generate a unique _id for each document
        "user_id": user_id,
        "session_id": session_id,
        "chat_history": {
            "query": query,
            "response": response,
            "timestamp": current_time_ist.strftime("%Y-%m-%d %H:%M:%S %Z%z")
        }
    }

    # Insert the document into the collection
    collection.insert_one(data_to_insert)


def get_latest_data(user_id, session_id, collection = get_collection(db_name = "nlp", collection_name = "nlp_package_v1")):
    ist_timezone = pytz.timezone('Asia/Kolkata')

    # Find the documents for the given user_id and session_id, sort by timestamp in descending order, and limit to 5
    cursor = collection.find(
        {"$and": [
            {"user_id": user_id},
            {"session_id": session_id}
        ]}
    ).sort("chat_history.timestamp", pymongo.DESCENDING).limit(5)


    latest_data = []

    for document in cursor:
        # timestamp_ist = datetime.strptime(document["chat_history"]["timestamp"], "%Y-%m-%d %H:%M:%S %Z")
        # timestamp_ist = timestamp_ist.replace(tzinfo=ist_timezone)

        data_entry = {
            "query": document["chat_history"]["query"],
            "response": document["chat_history"]["response"],
            "timestamp": document["chat_history"]["timestamp"]
        }

        latest_data.append(data_entry)

    return latest_data


def get_full_data(user_id, session_id, collection = get_collection(db_name = "nlp", collection_name = "nlp_package_v1")):
    ist_timezone = pytz.timezone('Asia/Kolkata')

    # Find the documents for the given user_id and session_id, sort by timestamp in descending order, and limit to 5
    cursor = collection.find(
        {"$and": [
            {"user_id": user_id},
            {"session_id": session_id}
        ]}
    ).sort("chat_history.timestamp", pymongo.DESCENDING)


    latest_data = []

    for document in cursor:
        # timestamp_ist = datetime.strptime(document["chat_history"]["timestamp"], "%Y-%m-%d %H:%M:%S %Z")
        # timestamp_ist = timestamp_ist.replace(tzinfo=ist_timezone)

        data_entry = {
            "query": document["chat_history"]["query"],
            "response": document["chat_history"]["response"]
        }

        latest_data.append(data_entry)

    return latest_data

