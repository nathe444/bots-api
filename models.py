from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class CustomKnowledge(Base):
    __tablename__ = "CustomKnowledge"
    __table_args__ = {"schema": "public"}
    
    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(Integer, index=True)  # Foreign key to Bots.BotID
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    file_path = Column(String(255))
    file_type = Column(String(50))
    file_size = Column(Integer)
    embedding_model = Column(String(100))
    is_embedded = Column(Boolean, default=False)
    embedding_status = Column(String(50), default='pending')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(Integer)
    document_metadata = Column(Text)  # Changed from metadata to document_metadata
    
    # Relationship with KnowledgeVector
    vectors = relationship("KnowledgeVector", back_populates="knowledge")

class KnowledgeVector(Base):
    __tablename__ = "KnowledgeVector"
    __table_args__ = {"schema": "public"}
    
    id = Column(Integer, primary_key=True, index=True)
    knowledge_id = Column(Integer, ForeignKey("public.CustomKnowledge.id"))
    chunk_index = Column(Integer)
    chunk_text = Column(Text)
    embedding = Column(Text)  # Serialized vector
    embedding_model = Column(String(100))
    dimensions = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship with CustomKnowledge
    knowledge = relationship("CustomKnowledge", back_populates="vectors")