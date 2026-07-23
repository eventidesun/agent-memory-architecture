import os
import chromadb
from dataclasses import dataclass

_ROOT = os.path.dirname(os.path.abspath(__file__))
client = chromadb.PersistentClient(path=os.path.join(_ROOT, "chroma_store"))


def get_or_create_collection(name):
    """One collection per corpus. All speakers in a corpus share it, so
    speaker identity is a metadata cue rather than a storage partition."""
    return client.get_or_create_collection(name=f"corpus_{name}")


def store_memory(corpus_name, user_message, agent_response, session_id,
                 speaker, memory_id, metadata=None,
                 user_label="Participant", agent_label="Lumen"):
    """Research path. Corpus, speaker, and memory ID are all explicit.

    Time is measured in sessions, so timestamp is the session number as a
    float, not wall-clock. Labels (type, salience, importance) are added by
    the batch labelling pass, not here.

    user_label / agent_label default to the values the frozen baseline was
    embedded with. Do not change the defaults.
    """
    collection = get_or_create_collection(corpus_name)

    base_metadata = {
        "speaker": speaker,
        "session_id": session_id,
        "timestamp": float(session_id),
        "labelled": False,
    }
    if metadata:
        base_metadata.update(metadata)

    collection.add(
        documents=[f"{user_label}: {user_message}\n{agent_label}: {agent_response}"],
        ids=[memory_id],
        metadatas=[base_metadata],
    )


def store_chat_memory(participant_id, user_message, agent_response, session_id):
    """Live-chat path: one collection per participant, auto-generated IDs.
    Separate from the research path, which uses a shared per-corpus store."""
    name = f"chat_{participant_id}"
    collection = get_or_create_collection(name)
    return store_memory(
        corpus_name=name,
        user_message=user_message,
        agent_response=agent_response,
        session_id=session_id,
        speaker=participant_id,
        memory_id=f"{participant_id}_{collection.count()}",
    )


def retrieve_memory(collection_name, current_message, n_results=3):
    collection = get_or_create_collection(collection_name)
    if collection.count() == 0:
        return ""
    results = collection.query(
        query_texts=[current_message],
        n_results=min(n_results, collection.count())
    )
    return "\n".join(results["documents"][0])


@dataclass
class ScoredMemory:
    memory_id: str
    text: str
    distance: float
    rank: int


def retrieve_scored(collection_name, current_message, n_results):
    collection = get_or_create_collection(collection_name)
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[current_message],
        n_results=min(n_results, collection.count()),
        include=["documents", "distances"]
    )

    ids = results["ids"][0]
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