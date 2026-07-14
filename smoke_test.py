# smoke_test.py
from memory import store_memory, retrieve_memory

PID = "smoke01"

store_memory(
    PID,
    "I love studying tide pools, they're my favorite part of the shore.",
    "That's wonderful — the intertidal zone is full of surprises.",
)

result = retrieve_memory(PID, "tide pools")
print("RETRIEVED:\n", result)