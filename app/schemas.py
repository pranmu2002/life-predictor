from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    is_admin: bool
    class Config:
        orm_mode = True

class SubmissionCreate(BaseModel):
    age: int
    bmi: float
    smoker: int
    stress_level: int
    chronic_disease: int
    diabetes: int
    income_level: int
    bp_systolic: int
    bp_diastolic: int
    vaccination_status: int
    pollution_index: int
    sleep_hours: float
    exercise_minutes: int
    alcohol_units: int
    diet_quality: int
    work_hours: int
    screen_time: int
    social_activity: int

class SubmissionResponse(BaseModel):
    id: int
    prediction: str
    class Config:
        orm_mode = True
