from sqlalchemy import Column, ForeignKey, Integer, String, Float
from db.base import Base

class Gyro(Base):
    __tablename__ = 'gyro'
    id = Column(Integer, primary_key=True)
    sessionid = Column(Integer, nullable=False)
    timestamp_in_microsecs = Column(Integer, nullable=False)
    gyro1 = Column(Float, nullable=False)
    gyro2 = Column(Float, nullable=False)
    gyro3 = Column(Float, nullable=False)
    rounded_scaled_ewma_gyroenergy = Column(Integer, nullable=False)
    scaled_gyroenergy = Column(Float, nullable=False)
    rounded_ewma_of_scaled_derivative_of_ewma_gyroenergy = Column(Integer, nullable=False)
    alpha_gyroenergy = Column(Float, nullable=False)
    min_gyroenergy_param = Column(Float, nullable=False)
    max_gyroenergy_param = Column(Float, nullable=False)
    alpha_ewma_derivative_of_gyroenergy = Column(Float, nullable=False)
    min_ewma_derivative_of_gyroenergy = Column(Float, nullable=False)
    max_ewma_derivative_of_gyroenergy = Column(Float, nullable=False)

