"""
All speakers share one collection, so speaker identity is a metadata cue
rather than a storage partition. Memory IDs are {speaker}_{index within
that speaker's exchanges}.
"""

import json
from collections import defaultdict

from memory import client, get_or_create_collection, store_memory
from run_labelling import label_corpus


def load_corpus(corpus_path: str, corpus_name: str, label: bool = True) -> dict:
    with open(corpus_path) as f:
        exchanges = json.load(f)

    try:
        client.delete_collection(name=f"corpus_{corpus_name}")
    except Exception:
        pass

    by_speaker = defaultdict(list)
    for e in exchanges:
        by_speaker[e["speaker"]].append(e)

    for pid, speaker_exchanges in by_speaker.items():
        for i, e in enumerate(speaker_exchanges):
            store_memory(
                corpus_name=corpus_name,
                user_message=e["user_message"],
                agent_response=e["agent_response"],
                session_id=e["session_id"],
                speaker=pid,
                memory_id=f"{pid}_{i}",
            )

    col = get_or_create_collection(corpus_name)
    for pid, speaker_exchanges in by_speaker.items():
        for i, e in enumerate(speaker_exchanges):
            mem_id = f"{pid}_{i}"
            meta = dict(col.get(ids=[mem_id], include=["metadatas"])["metadatas"][0])
            log = e.get("presentation_log", [float(e["session_id"])])
            meta["presentation_log"] = json.dumps(log)
            col.update(ids=[mem_id], metadatas=[meta])

    counts = label_corpus(corpus_name, verbose=False) if label else {}

    return {"exchanges": len(exchanges),
            "speakers": sorted(by_speaker),
            "collection": f"corpus_{corpus_name}",
            "labelling": counts}


if __name__ == "__main__":
    print(load_corpus("benchmark/corpus_primary.json", corpus_name="primary"))