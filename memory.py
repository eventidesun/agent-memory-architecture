import chromadb
from dataclasses import dataclass          # ← NEW: add this import at the top

client = chromadb.Client()

def get_or_create_collection(participant_id):
    return client.get_or_create_collection(name=f"participant_{participant_id}")

def store_memory(participant_id, user_message, lumen_response):
    collection = get_or_create_collection(participant_id)
    collection.add(
        documents=[f"Participant: {user_message}\nLumen: {lumen_response}"],
        ids=[f"{participant_id}_{collection.count()}"]
    )

def retrieve_memory(participant_id, current_message, n_results=3):
    # ── your existing function, UNCHANGED ──
    collection = get_or_create_collection(participant_id)
    if collection.count() == 0:
        return ""
    results = collection.query(
        query_texts=[current_message],
        n_results=min(n_results, collection.count())
    )
    memories = results["documents"][0]
    return "\n".join(memories)


# ══════════════════════════════════════════════════════════
# NEW below this line — the instrumented retrieval
# ══════════════════════════════════════════════════════════

@dataclass
class ScoredMemory:
    memory_id: str
    text: str
    distance: float
    rank: int

def retrieve_scored(participant_id, current_message, n_results):
    collection = get_or_create_collection(participant_id)
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[current_message],
        n_results=min(n_results, collection.count()),
        include=["documents", "distances"]     # ← ask ChromaDB for distances explicitly
    )

    ids       = results["ids"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]

    scored = []
    for rank, (mem_id, text, dist) in enumerate(zip(ids, documents, distances), start=1):
        scored.append(ScoredMemory(
            memory_id=mem_id,
            text=text,
            distance=dist,
            rank=rank
        ))
    return scored
