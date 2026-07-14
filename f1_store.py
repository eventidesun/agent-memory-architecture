# f1_store.py
from memory import store_memory, get_or_create_collection

PID = "persist01"
store_memory(
    PID,
    "My favorite depth is 200 meters. Remember that.",
    "Noted — 200 meters. I won't forget.",
)
print("Stored. Collection count:", get_or_create_collection(PID).count())
print("Exiting process now.")
