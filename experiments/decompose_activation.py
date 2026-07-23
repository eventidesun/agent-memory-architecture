"""Decompose A = B + S + noise on the corpus, to see which term carries
the variance. Uses the same query distribution as the weight derivation
(the corpus's own user messages) so the two are directly comparable.
Diagnostic only; writes nothing."""

import json
import statistics

from activation.base_level import base_level
from activation.spreading import spreading
from activation.params import W, LAMBDA, NOISE_SIGMA
from memory import get_or_create_collection

CURRENT_TIME = 6.0


def user_message(document):
    first_line = document.split("\n")[0]
    return first_line.split(": ", 1)[1] if ": " in first_line else first_line


def run(corpus_name="primary"):
    col = get_or_create_collection(corpus_name)
    stored = col.get(include=["documents", "metadatas", "embeddings"])

    mem_embeddings = [list(e) for e in stored["embeddings"]]
    metas = stored["metadatas"]
    speakers = [m["speaker"] for m in metas]

    queries = [user_message(d) for d in stored["documents"]]
    query_embeddings = [list(e) for e in col._embedding_function(queries)]

    bs, ss, s_utt, s_spk = [], [], [], []

    for q_emb, q_speaker in zip(query_embeddings, speakers):
        for meta, m_emb in zip(metas, mem_embeddings):
            log = json.loads(meta["presentation_log"])
            b = base_level(log, CURRENT_TIME, float(meta["salience"]), LAMBDA)
            s = spreading(q_emb, m_emb, meta["speaker"], q_speaker, W)
            utt_only = spreading(q_emb, m_emb, meta["speaker"], q_speaker,
                                 {"utterance": W["utterance"], "speaker": 0.0})
            bs.append(b)
            ss.append(s)
            s_utt.append(utt_only)
            s_spk.append(s - utt_only)

    def report(name, vals):
        print(f"{name:22} mean={statistics.mean(vals):7.3f}  "
              f"sd={statistics.stdev(vals):6.3f}  "
              f"range=[{min(vals):.3f}, {max(vals):.3f}]")

    print(f"corpus = {corpus_name}, pairs = {len(bs)}, noise sigma = {NOISE_SIGMA}\n")
    report("B (base-level)", bs)
    report("S (total spreading)", ss)
    report("  S from utterance", s_utt)
    report("  S from speaker", s_spk)


if __name__ == "__main__":
    run()