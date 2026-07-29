"""
MongoDB connection helper.

Uses a local MongoDB instance by default (mongodb://localhost:27017) or
MongoDB Atlas free tier if MONGO_URI is set in .env — either works with
zero code changes, since it's just a connection string.
"""

import os
from functools import lru_cache
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://amareshotta_db_user:G3yjT5R6vPXJ8Ixm@citeminddb.vimztad.mongodb.net/")
DB_NAME = os.getenv("MONGO_DB_NAME", "citemind")


@lru_cache
def get_client():
    return MongoClient(MONGO_URI)


def get_db():
    """Returns the CiteMind database. Collections used:
    - notebooks     (name, persona config, provider preference)
    - documents     (filename, chunk metadata: doc_id/page/offset)
    - chat_history   (question, answer, citations, timestamp)
    - eval_runs      (score snapshots over time)
    """
    client = get_client()
    return client[DB_NAME]
