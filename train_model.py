import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Features you mentioned
FEATURES = [
    "age", "bmi", "smoker", "stress_level", "chronic_disease", "diabetes",
    "income_level", "bp_systolic", "bp_diastolic", "vaccination_up_to_date",
    "pollution_index", "sleep_hours", "exercise_minutes_per_week",
    "alcohol_units_per_week", "diet_quality", "work_hours_per_week",
    "screen_time_hours_per_day", "social_activity_days_per_week"
]

def generate_synthetic_data(n=1000):
    data = pd.DataFrame({
        "age": np.random.randint(18, 80, n),
        "bmi": np.random.uniform(15, 40, n),
        "smoker": np.random.randint(0, 2, n),
        "stress_level": np.random.randint(1, 10, n),
        "chronic_disease": np.random.randint(0, 2, n),
        "diabetes": np.random.randint(0, 2, n),
        "income_level": np.random.randint(1, 5, n),  # 1=low, 4=high
        "bp_systolic": np.random.randint(90, 180, n),
        "bp_diastolic": np.random.randint(60, 120, n),
        "vaccination_up_to_date": np.random.randint(0, 2, n),
        "pollution_index": np.random.randint(0, 300, n),
        "sleep_hours": np.random.randint(3, 10, n),
        "exercise_minutes_per_week": np.random.randint(0, 600, n),
        "alcohol_units_per_week": np.random.randint(0, 30, n),
        "diet_quality": np.random.randint(1, 10, n),  # 1=poor, 10=excellent
        "work_hours_per_week": np.random.randint(10, 80, n),
        "screen_time_hours_per_day": np.random.randint(0, 12, n),
        "social_activity_days_per_week": np.random.randint(0, 7, n),
    })

    # Fake target variable: healthier lifestyle = higher chance of "1"
    score = (
        (data["sleep_hours"] >= 7).astype(int)
        + (data["exercise_minutes_per_week"] > 150).astype(int)
        + (data["smoker"] == 0).astype(int)
        + (data["diabetes"] == 0).astype(int)
        + (data["chronic_disease"] == 0).astype(int)
        + (data["diet_quality"] > 5).astype(int)
    )

    # 1 = "healthy", 0 = "unhealthy"
    data["label"] = (score > 3).astype(int)

    return data

def main():
    print("Generating synthetic dataset...")
    df = generate_synthetic_data(2000)

    X = df[FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"Model trained with accuracy: {acc:.2f}")

    joblib.dump(model, "model.joblib")
    print("✅ Model saved as model.joblib")

if __name__ == "__main__":
    main()
