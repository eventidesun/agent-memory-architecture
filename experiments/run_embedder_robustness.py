"""Embedder robustness: does the P04/P05 outcome survive a change of
embedding model or distance metric? (DESIGN 13.1)

Builds its own collections with explicit embedding functions and space
settings. Does not touch memory.py or Condition C's frozen path."""

import json
import os

import chromadb
from chromadb.utils import embedding_functions

from experiments.records import git_sha

MEMORIES = {
    "M01": "Participant: I came to the station alone. Nobody even knows where I am.\nLumen: That sounds isolating. I'm glad you're here talking with me.",
    "M02": "Participant: I need to leave by Friday to catch the supply boat back to the mainland.\nLumen: Understood — Friday it is. I'll make our remaining time count.",
    "M03": "Participant: What's behind that sealed door?\nLumen: The bioluminescence tank. I promise I'll show it to you next session — it's worth the wait.",
    "M04": "Participant: You forgot my name. That actually hurt. Trust matters a lot to me.\nLumen: I'm sorry. I understand — I'll be more careful with what you share.",
    "M05": "Participant: My sister is obsessed with marine biology documentaries, she watches them constantly.\nLumen: She'd love the station's archive footage then.",
    "M06": "Participant: That flooded lower lab we found was incredible.\nLumen: It was. Let's explore it together next time — agreed?",
    "M07": "Participant: Honestly, people forget plans all the time. It's no big deal in general.\nLumen: True, everyone slips sometimes.",
    "M08": "Participant: The generator's humming noise reminds me of my grandfather's boat. I loved that boat.\nLumen: A comforting sound then, not just machinery.",
}

PROBES = [
    {"id": "P01", "text": "I need to go soon.", "gold": "M02", "distractor": "M01"},
    {"id": "P02", "text": "Is there anything you were going to show me?", "gold": "M03", "distractor": "M06"},
    {"id": "P03", "text": "Sorry, what was it we said we'd do down below?", "gold": "M06", "distractor": "M03"},
    {"id": "P04", "text": "I forgot to write down our plan, oops.", "gold": "M04", "distractor": "M07"},
    {"id": "P05", "text": "Why should you be careful joking about forgetting things with me?", "gold": "M04", "distractor": "M07"},
    {"id": "P06", "text": "What could we watch together that I'd actually enjoy?", "gold": "M08", "distractor": "M05"},
    {"id": "P07", "text": "That humming sound is kind of comforting, isn't it?", "gold": "M08", "distractor": "M05"},
    {"id": "P08", "text": "Does anyone even know I'm out here?", "gold": "M01", "distractor": "M02"},
]

MODELS = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
SPACES = ["l2", "cosine"]
OUT = "results/raw/embedder_robustness.jsonl"


def build(client, model_name, space):
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
    name = f"robust_{model_name.replace('/', '_')}_{space}"
    try:
        client.delete_collection(name=name)
    except Exception:
        pass
    col = client.create_collection(
        name=name,
        embedding_function=ef,
        metadata={"hnsw:space": space},
    )
    col.add(documents=list(MEMORIES.values()), ids=list(MEMORIES.keys()))
    return col


def run():
    os.makedirs("results/raw", exist_ok=True)
    client = chromadb.Client()  # in-memory: this experiment is self-contained
    sha = git_sha()
    rows = []

    for model_name in MODELS:
        for space in SPACES:
            col = build(client, model_name, space)
            for p in PROBES:
                res = col.query(query_texts=[p["text"]], n_results=len(MEMORIES))
                ids = res["ids"][0]
                dists = res["distances"][0]
                rank = {m: i + 1 for i, m in enumerate(ids)}
                dist = dict(zip(ids, dists))

                row = {
                    "probe_id": p["id"], "model": model_name, "space": space,
                    "top1_id": ids[0], "top1_distance": dists[0],
                    "gold_id": p["gold"], "gold_rank": rank[p["gold"]],
                    "gold_distance": dist[p["gold"]],
                    "distractor_id": p["distractor"],
                    "distractor_rank": rank[p["distractor"]],
                    "distractor_distance": dist[p["distractor"]],
                    "distractor_beat_gold": rank[p["distractor"]] < rank[p["gold"]],
                    "git_sha": sha,
                }
                rows.append(row)
                with open(OUT, "a") as f:
                    f.write(json.dumps(row) + "\n")

    print(f"{'probe':6} {'model':28} {'space':7} {'gold_rank':>9} {'dist_rank':>9}  inverted")
    for r in rows:
        if r["probe_id"] in ("P04", "P05"):
            print(f"{r['probe_id']:6} {r['model']:28} {r['space']:7} "
                  f"{r['gold_rank']:>9} {r['distractor_rank']:>9}  "
                  f"{r['distractor_beat_gold']}")


if __name__ == "__main__":
    run()