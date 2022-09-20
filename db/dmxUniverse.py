
from sqlalchemy import Column, ForeignKey, Integer, String
from db.base import Base

class DMXUniverse(Base):
    __tablename__ = 'idToOscAddress'
    id = Column(Integer, primary_key=True)
    reference = Column(Integer, nullable=False)
    address = Column(String(250), nullable=False)
