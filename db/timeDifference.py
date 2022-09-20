
from sqlalchemy import Column, ForeignKey, Integer, String
from db.base import Base


class TimeDifference(Base):
    __tablename__ = 'timedifference'
    id = Column(Integer, primary_key=True)
    time = Column(Integer, nullable=False)
    name = Column(String(250), nullable=False)

# # class Address(Base):
# #     __tablename__ = 'address'
# #     # Here we define columns for the table address.
# #     # Notice that each column is also a normal Python instance attribute.
# #     id = Column(Integer, primary_key=True)
# #     street_name = Column(String(250))
# #     street_number = Column(String(250))
# #     post_code = Column(String(250), nullable=False)
# #     person_id = Column(Integer, ForeignKey('person.id'))
# #     person = relationship(Person)

# # Create an engine that stores data in the local directory's
# # sqlalchemy_example.db file.
# engine = create_engine('sqlite:///sqlalchemy_example.db')

# # Create all tables in the engine. This is equivalent to "Create Table"
# # statements in raw SQL.
# Base.metadata.create_all(engine)