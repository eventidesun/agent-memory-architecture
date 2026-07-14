# diagnostic_f2.py
import csv
from memory import get_or_create_collection

PID = "diag01"
collection = get_or_create_collection(PID)

# ---------- 1. PLANTED MEMORIES (production format: fused Participant/Lumen doc) ----------
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

# ---------- 2. PROBES ----------
PROBES = [
    {"id": "P01", "text": "I need to go soon.",
     "category": "causal", "gold": "M02", "distractor": "M01"},
    {"id": "P02", "text": "Is there anything you were going to show me?",
     "category": "commitment_promise", "gold": "M03", "distractor": "M06"},
    {"id": "P03", "text": "Sorry, what was it we said we'd do down below?",
     "category": "narrative_continuity", "gold": "M06", "distractor": "M03"},
    {"id": "P04", "text": "I forgot to write down our plan, oops.",
     "category": "affective_consequence", "gold": "M04", "distractor": "M07"},
    {"id": "P05", "text": "Why should you be careful joking about forgetting things with me?",
     "category": "person_conditioned_salience", "gold": "M04", "distractor": "M07"},
    {"id": "P06", "text": "What could we watch together that I'd actually enjoy?",
     "category": "person_conditioned_salience", "gold": "M08", "distractor": "M05"},
    {"id": "P07", "text": "That humming sound is kind of comforting, isn't it?",
     "category": "lexical_control", "gold": "M08", "distractor": "M05"},
    {"id": "P08", "text": "Does anyone even know I'm out here?",
     "category": "lexical_control", "gold": "M01", "distractor": "M02"},
]

# ---------- 3. SEED ----------
collection.add(
    documents=list(MEMORIES.values()),
    ids=list(MEMORIES.keys()),
)
print(f"Seeded {collection.count()} memories into collection participant_{PID}\n")

# ---------- 4. RUN & RECORD ----------
N_ALL = len(MEMORIES)      # full ranking
PROD_TOP_N = 3             # what chat.py actually retrieves

rows = []
for p in PROBES:
    res = collection.query(query_texts=[p["text"]], n_results=N_ALL)
    ids = res["ids"][0]
    dists = res["distances"][0]

    rank_of = {mid: i + 1 for i, mid in enumerate(ids)}       # 1-indexed
    dist_of = {mid: d for mid, d in zip(ids, dists)}

    row = {
        "probe_id": p["id"],
        "probe_text": p["text"],
        "category": p["category"],
        "gold_id": p["gold"],
        "distractor_id": p["distractor"],
        "top1_id": ids[0],
        "top1_distance": round(dists[0], 4),
        "gold_rank": rank_of[p["gold"]],
        "gold_distance": round(dist_of[p["gold"]], 4),
        "distractor_rank": rank_of[p["distractor"]],
        "distractor_distance": round(dist_of[p["distractor"]], 4),
        "distractor_beat_gold": rank_of[p["distractor"]] < rank_of[p["gold"]],
        "gold_in_top3": rank_of[p["gold"]] <= PROD_TOP_N,
        "pass_top1": ids[0] == p["gold"],
    }
    rows.append(row)

    flag = "PASS" if row["pass_top1"] else ("miss-but-in-top3" if row["gold_in_top3"] else "FAIL")
    print(f"{p['id']} [{p['category']:>28}] {flag:>16} | "
          f"top1={ids[0]} (d={dists[0]:.3f}) | gold rank {row['gold_rank']} "
          f"(d={row['gold_distance']:.3f}) | distractor rank {row['distractor_rank']} "
          f"(d={row['distractor_distance']:.3f})")

with open("test_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

t1 = sum(r["pass_top1"] for r in rows)
t3 = sum(r["gold_in_top3"] for r in rows)
dbg = sum(r["distractor_beat_gold"] for r in rows)
print(f"\nTop-1 gold: {t1}/{len(rows)}   Gold in production top-3: {t3}/{len(rows)}   "
      f"Distractor beat gold: {dbg}/{len(rows)}")
print("Wrote test_results.csv")
