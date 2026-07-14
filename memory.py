import chromadb

client = chromadb.Client() # intializing an in-memory chromadb client

# Chromadb is a vector database, so data is organized into "collections"
def get_or_create_collection(participant_id):
    return client.get_or_create_collection(name=f"participant_{participant_id}")

def store_memory(participant_id, user_message, lumen_response):
    collection = get_or_create_collection(participant_id)
    collection.add(
        documents=[f"Participant: {user_message}\nLumen: {lumen_response}"],
        ids=[f"{participant_id}_{collection.count()}"]
    )

def retrieve_memory(participant_id, current_message, n_results=3):
    collection = get_or_create_collection(participant_id)
    if collection.count() == 0:
        return ""
    results = collection.query(
        query_texts=[current_message],
        n_results=min(n_results, collection.count())
    )
    memories = results["documents"][0]
    return "\n".join(memories)
