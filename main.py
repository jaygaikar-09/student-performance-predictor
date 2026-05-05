import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
model = joblib.load("best_model.pkl")

print("Student Exam Score Predictor")
study_hours = float(input("Study Hours: "))
attendance = float(input("Attendance Percentage: "))
mental_health = int(input("Mental Health (1-10): "))
sleep_hours = float(input("Sleep Hours: "))
part_time_job = input("Part-Time Job (Yes/No): ")

ptj_encoded = 1 if part_time_job.lower() == "yes" else 0

data = np.array([[study_hours, attendance, mental_health, sleep_hours, ptj_encoded]])

prediction = model.predict(data)[0]

prediction = max(0, min(100, prediction))

print(f"\n Predicted Exam Score: {prediction:.2f}")