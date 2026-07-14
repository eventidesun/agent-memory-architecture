# f1_check.py
from memory import retrieve_memory, get_or_create_collection

PID = "persist01"
print("Collection count on fresh process:", get_or_create_collection(PID).count())
print("RETRIEVED:", repr(retrieve_memory(PID, "favorite depth")))
