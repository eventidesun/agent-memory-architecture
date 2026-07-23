"""Wipe and rebuild collections from a corpus file. Deterministic:
same corpus in, same store state out. Run before any experiment."""

import json
from collections import defaultdict

from memory import client, get_or_create_collection, store_memory
from run_labelling import label_corpus


def load_corpus(corpus_path: str, label: bool = True) -> dict:
    with open(corpus_path) as f:
        exchanges = json.load(f)

    by_speaker = defaultdict(list)
    for e in exchanges:
        by_speaker[e["speaker"]].append(e)

    counts = {}
    for pid, speaker_exchanges in by_speaker.items():
        try:
            client.delete_collection(name=f"participant_{pid}")
        except Exception:
            pass

        for e in speaker_exchanges:
            store_memory(pid, e["user_message"], e["agent_response"],
                         e["session_id"])

        col = get_or_create_collection(pid)
        for i, e in enumerate(speaker_exchanges):
            mem_id = f"{pid}_{i}"
            meta = dict(col.get(ids=[mem_id], include=["metadatas"])["metadatas"][0])
            log = e.get("presentation_log", [float(e["session_id"])])
            meta["presentation_log"] = json.dumps(log)
            col.update(ids=[mem_id], metadatas=[meta])

        if label:
            counts[pid] = label_corpus(pid, verbose=False)

    return {"exchanges": len(exchanges),
            "speakers": sorted(by_speaker),
            "labelling": counts}


if __name__ == "__main__":
    print(load_corpus("benchmark/corpus_lumen.json"))