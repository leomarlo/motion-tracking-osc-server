from sqlalchemy import Column, ForeignKey, Integer, String, Float
from db.base import Base

class RecordingSession(Base):
    __tablename__ = 'recordingsession'
    id = Column(Integer, primary_key=True)
    starting_timestamp_in_microsecs = Column(Integer, nullable=False)
    ip_address = Column(String(250), nullable=False)
    port = Column(Integer, nullable=False)
    host_name = Column(String(500), nullable=False)
