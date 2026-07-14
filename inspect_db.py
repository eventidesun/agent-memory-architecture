# inspect_db.py
from memory import store_memory, get_or_create_collection

PID = "inspect01"

store_memory(
    PID,
    "I came to the station alone. Nobody knows where I am.",
    "That sounds isolating. I'm glad you're talking to me.",
)

collection = get_or_create_collection(PID)
data = collection.get(include=["documents", "metadatas", "embeddings"])

print("IDs:      ", data["ids"])
print("Documents:", data["documents"])
print("Metadatas:", data["metadatas"])
emb = data["embeddings"][0]
print("Embedding dimensions:", len(emb))
print("First 5 dims:", [round(float(x), 4) for x in emb[:5]])
