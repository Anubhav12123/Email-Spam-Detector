# api/models.py
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.sql import func
from db import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    sender = Column(String(320), nullable=True)
    subject = Column(String(500), nullable=True)
    text = Column(Text, nullable=False)
    label = Column(Boolean, nullable=False)
    probability = Column(Float, nullable=False)
    model_name = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
