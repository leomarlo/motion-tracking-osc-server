from sqlalchemy import Column, ForeignKey, Integer, String, Float
from db.base import Base

class PoseLandmarks(Base):
    __tablename__ = 'poselandmarks'
    landmarkid = Column(Integer, primary_key=True)
    name  = Column(String(250), nullable=False)