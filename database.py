from sqlalchemy import create_engine, Column, Integer, String, DateTime, func
from sqlalchemy.orm import declarative_base, sessionmaker

engine = create_engine("sqlite:///chat_history.db")
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(String)
    answer = Column(String)
    document_name = Column(String)
    created_at = Column(DateTime, server_default=func.now())

# Creates the table if it doesn't already exist
Base.metadata.create_all(bind=engine)