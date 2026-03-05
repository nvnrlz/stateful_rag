# stateful_rag
A drop-in Python module (or a custom Retriever plugin for frameworks like LangChain/LlamaIndex). On Turn 1, it queries the main vector DB and saves the retrieved "sub-graph" to a PostgreSQL session table. On Turn 2+, it routes the query only to the cached PostgreSQL sub-graph, executing your "constrained re-analysis."
