"""ENT AI-Receptionist backend.

A multi-turn, retrieval-grounded triage/receptionist service built on top of the
hardened ``stateful_rag`` caching layer. It conducts a back-and-forth
conversation, asks clarifying questions to localise a patient's ENT complaint,
grounds its statements in the licensed reference corpus, enforces red-flag
escalation, and is strictly framed as an educational test tool — not a
diagnostic device.
"""

__version__ = "0.1.0"
