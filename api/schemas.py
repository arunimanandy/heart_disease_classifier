from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    age: float = Field(..., example=55)
    sex: float = Field(..., example=1)
    cp: float = Field(..., example=4)
    trestbps: float = Field(..., example=140)
    chol: float = Field(..., example=250)
    fbs: float = Field(..., example=0)
    restecg: float = Field(..., example=1)
    thalach: float = Field(..., example=150)
    exang: float = Field(..., example=0)
    oldpeak: float = Field(..., example=1.2)
    slope: float = Field(..., example=2)
    ca: float = Field(..., example=0)
    thal: float = Field(..., example=3)
