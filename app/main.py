from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, Field
from io import BytesIO

import joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from . import models, auth, database


# -------------------- App & CORS --------------------
app = FastAPI(title="Life Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:5500", "http://127.0.0.1:5500", "https://lifepredictor.netlify.app"],
          # ✅ change to your Netlify domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],          # ✅ allows Authorization header
)


# -------------------- DB Init --------------------
models.Base.metadata.create_all(bind=database.engine)


# -------------------- Pydantic Schemas --------------------
class RegisterIn(BaseModel):
    name: str = Field(..., min_length=1)
    email: EmailStr
    password: str = Field(..., min_length=4)


class SubmissionIn(BaseModel):
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


class SubmissionOut(BaseModel):
    id: int
    user_id: int
    prediction: str

    class Config:
        from_attributes = True


# -------------------- Load ML Model --------------------
MODEL_PATH = "model.joblib"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"⚠️ WARNING: Could not load model at '{MODEL_PATH}': {e}")

FEATURE_ORDER = [
    "age", "bmi", "smoker", "stress_level", "chronic_disease", "diabetes",
    "income_level", "bp_systolic", "bp_diastolic", "vaccination_status",
    "pollution_index", "sleep_hours", "exercise_minutes", "alcohol_units",
    "diet_quality", "work_hours", "screen_time", "social_activity"
]


# -------------------- Startup: Seed Admin --------------------
@app.on_event("startup")
def seed_admin():
    db = next(database.get_db())
    admin = db.query(models.User).filter(models.User.email == "swagatoroy2002@gmail.com").first()
    if not admin:
        hashed = auth.get_password_hash("moyu2002")
        admin = models.User(
            name="Admin",
            email="swagatoroy2002@gmail.com",
            hashed_password=hashed,
            is_admin=True
        )
        db.add(admin)
        db.commit()
    db.close()


# -------------------- Auth Routes --------------------
@app.post("/register", status_code=201)
def register(payload: RegisterIn, db: Session = Depends(database.get_db)):
    exists = db.query(models.User).filter(models.User.email == payload.email).first()
    if exists:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pw = auth.get_password_hash(payload.password)
    user = models.User(
        name=payload.name,
        email=payload.email,
        hashed_password=hashed_pw,
        is_admin=False
    )
    db.add(user)
    db.commit()

    return {"success": True, "message": "✅ User registered successfully. You can now log in."}


@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="❌ Invalid email or password")

    token = auth.create_access_token({"sub": user.email})
    return {
        "success": True,
        "message": "✅ Login successful",
        "access_token": token,
        "token_type": "bearer",
        "is_admin": user.is_admin
    }


@app.get("/me")
def me(current_user: models.User = Depends(auth.get_current_user)):
    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
        "is_admin": current_user.is_admin
    }


# -------------------- Prediction & Submissions --------------------
@app.post("/submit", response_model=SubmissionOut)
def submit(
    payload: SubmissionIn,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    if model is None:
        raise HTTPException(status_code=500, detail="⚠️ Model not loaded on server")

    numeric_features = [[
        int(payload.age), float(payload.bmi), int(payload.smoker),
        int(payload.stress_level), int(payload.chronic_disease), int(payload.diabetes),
        int(payload.income_level), int(payload.bp_systolic), int(payload.bp_diastolic),
        int(payload.vaccination_status), int(payload.pollution_index), float(payload.sleep_hours),
        int(payload.exercise_minutes), int(payload.alcohol_units), int(payload.diet_quality),
        int(payload.work_hours), int(payload.screen_time), int(payload.social_activity),
    ]]

    pred = model.predict(numeric_features)[0]
    label = "Healthy" if int(pred) == 1 else "Unhealthy"

    sub = models.Submission(
        user_id=current_user.id,
        **payload.dict(),
        smoker=bool(payload.smoker),
        chronic_disease=bool(payload.chronic_disease),
        diabetes=bool(payload.diabetes),
        vaccination_status=bool(payload.vaccination_status),
        prediction=label
    )
    db.add(sub)
    db.commit()
    db.refresh(sub)

    return {"id": sub.id, "user_id": sub.user_id, "prediction": sub.prediction}


# -------------------- Admin Only: View Submissions --------------------
@app.get("/admin/submissions")
def admin_submissions(db: Session = Depends(database.get_db), _: models.User = Depends(auth.get_current_admin)):
    return db.query(models.Submission).all()


# -------------------- PDF Download --------------------
@app.get("/download/{submission_id}")
def download(
    submission_id: int,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    sub = db.query(models.Submission).filter(models.Submission.id == submission_id).first()
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    if (not current_user.is_admin) and (sub.user_id != current_user.id):
        raise HTTPException(status_code=403, detail="Not authorized to download this report")

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    story = [Paragraph("🩺 Prediction Report", styles["Title"]), Spacer(1, 12)]
    fields = [
        "id","user_id","age","bmi","smoker","stress_level","chronic_disease","diabetes",
        "income_level","bp_systolic","bp_diastolic","vaccination_status","pollution_index",
        "sleep_hours","exercise_minutes","alcohol_units","diet_quality","work_hours",
        "screen_time","social_activity","prediction"
    ]
    for k in fields:
        v = getattr(sub, k)
        story.append(Paragraph(f"<b>{k.capitalize()}</b>: {v}", styles["Normal"]))
        story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)

    headers = {"Content-Disposition": f'attachment; filename="submission_{submission_id}.pdf"'}
    return StreamingResponse(buffer, media_type="application/pdf", headers=headers)
