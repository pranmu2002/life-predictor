from sqlalchemy import Column, Integer, String, Boolean, Float, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False)
    submissions = relationship("Submission", back_populates="owner")

class Submission(Base):
    __tablename__ = "submissions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    age = Column(Integer)
    bmi = Column(Float)
    smoker = Column(Boolean)
    stress_level = Column(Integer)
    chronic_disease = Column(Boolean)
    diabetes = Column(Boolean)
    income_level = Column(Integer)
    bp_systolic = Column(Integer)
    bp_diastolic = Column(Integer)
    vaccination_status = Column(Boolean)
    pollution_index = Column(Integer)
    sleep_hours = Column(Float)
    exercise_minutes = Column(Integer)
    alcohol_units = Column(Integer)
    diet_quality = Column(Integer)
    work_hours = Column(Integer)
    screen_time = Column(Integer)
    social_activity = Column(Integer)
    prediction = Column(String)
    owner = relationship("User", back_populates="submissions")
