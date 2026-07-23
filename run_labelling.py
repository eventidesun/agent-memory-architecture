"""Batch labelling pass. Reads memories from the store, labels the unlabelled
ones, writes labels back."""

from labelling import label_memory
from memory import get_or_create_collection


def label_corpus(participant_id, force=False, verbose=True):
    collection = get_or_create_collection(participant_id)
    stored = collection.get(include=["documents", "metadatas"])

    ids = stored["ids"]
    documents = stored["documents"]
    metadatas = stored["metadatas"]

    labelled_count = 0
    skipped_count = 0

    for mem_id, text, meta in zip(ids, documents, metadatas):
        meta = meta or {}

        if meta.get("labelled") and not force:
            skipped_count += 1
            continue

        labels = label_memory(text, meta.get("speaker", participant_id))

        merged = dict(meta)
        merged.update({
            "type": labels["type"],
            "salience": labels["salience"],
            "importance": labels["importance"],
            "labelled": True,
        })

        collection.update(ids=[mem_id], metadatas=[merged])
        labelled_count += 1

        if verbose:
            print(f"{mem_id}  {labels['type']:20s} "
                  f"sal={labels['salience']:.2f} imp={labels['importance']:.2f}")

    return {"labelled": labelled_count, "skipped": skipped_count}