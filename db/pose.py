from sqlalchemy import Column, ForeignKey, Integer, String, Float
from db.base import Base

class MediapipePose(Base):
    __tablename__ = 'mediapipePose'
    id = Column(Integer, primary_key=True)
    sessionid = Column(Integer, nullable=False)
    timestamp = Column(Integer, nullable=False)
    x = Column(Integer, nullable=False)
    y = Column(Integer, nullable=False)
    z = Column(Integer, nullable=False)
    landmarkid = Column(Integer, nullable=False)



class PoseLandmarks(Base):
    __tablename__ = 'poselandmarks'
    landmarkid = Column(Integer, primary_key=True)
    name  = Column(String(250), nullable=False)