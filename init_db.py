from sqlalchemy import create_engine, text
from stateful_rag.models import Base

DB_URL = "postgresql+psycopg://rag_user:rag_password@localhost:5433/rag_state"


def init_database():
    print("Connecting to PostgreSQL Docker Container...")
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        print("Enabling pgvector extension for high-dimensional similarity search...")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()

    print("Creating 'sessions' and 'cached_nodes' relational tables...")
    Base.metadata.create_all(engine)
    print("✅ Database initialized successfully!")


if __name__ == "__main__":
    init_database()
