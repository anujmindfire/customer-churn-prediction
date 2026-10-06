"""Pydantic data schemas for Customer Churn Prediction API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CustomerDataRequest(BaseModel):
    """Input request schema containing raw customer features for churn prediction."""

    customerID: str | None = Field(
        default=None,
        description="Optional unique customer identifier.",
        examples=["7590-VHVEG"],
    )
    gender: Literal["Male", "Female"] = Field(
        ...,
        description="Customer gender.",
        examples=["Female"],
    )
    SeniorCitizen: Literal[0, 1] = Field(
        ...,
        description="Whether customer is a senior citizen (1) or not (0).",
        examples=[0],
    )
    Partner: Literal["Yes", "No"] = Field(
        ...,
        description="Whether the customer has a partner.",
        examples=["Yes"],
    )
    Dependents: Literal["Yes", "No"] = Field(
        ...,
        description="Whether the customer has dependents.",
        examples=["No"],
    )
    tenure: int = Field(
        ...,
        ge=0,
        description="Number of months the customer has stayed with the company.",
        examples=[1],
    )
    PhoneService: Literal["Yes", "No"] = Field(
        ...,
        description="Whether customer has a phone service.",
        examples=["No"],
    )
    MultipleLines: Literal["No", "Yes", "No phone service"] = Field(
        ...,
        description="Whether the customer has multiple phone lines.",
        examples=["No phone service"],
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ...,
        description="Customer's internet service provider.",
        examples=["DSL"],
    )
    OnlineSecurity: Literal["No", "Yes", "No internet service"] = Field(
        ...,
        description="Whether customer has online security service.",
        examples=["No"],
    )
    OnlineBackup: Literal["No", "Yes", "No internet service"] = Field(
        ...,
        description="Whether customer has online backup service.",
        examples=["Yes"],
    )
    DeviceProtection: Literal["No", "Yes", "No internet service"] = Field(
        ...,
        description="Whether customer has device protection service.",
        examples=["No"],
    )
    TechSupport: Literal["No", "Yes", "No internet service"] = Field(
        ...,
        description="Whether customer has tech support service.",
        examples=["No"],
    )
    StreamingTV: Literal["No", "Yes", "No internet service"] = Field(
        ...,
        description="Whether customer has streaming TV service.",
        examples=["No"],
    )
    StreamingMovies: Literal["No", "Yes", "No internet service"] = Field(
        ...,
        description="Whether customer has streaming movies service.",
        examples=["No"],
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ...,
        description="Contract term of the customer.",
        examples=["Month-to-month"],
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ...,
        description="Whether the customer has paperless billing enabled.",
        examples=["Yes"],
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(
        ...,
        description="Customer payment method.",
        examples=["Electronic check"],
    )
    MonthlyCharges: float = Field(
        ...,
        ge=0.0,
        description="Amount charged to the customer monthly.",
        examples=[29.85],
    )
    TotalCharges: float | str = Field(
        ...,
        description="Total charges accumulated (numeric float or string; whitespace converted to 0.0).",
        examples=[29.85],
    )

    @field_validator("TotalCharges")
    @classmethod
    def validate_total_charges(cls, v: float | str) -> float | str:
        """Validate and clean TotalCharges values."""
        if isinstance(v, (int, float)):
            if v < 0:
                raise ValueError("TotalCharges cannot be negative.")
            return float(v)
        if isinstance(v, str):
            cleaned = v.strip()
            if cleaned == "":
                return 0.0
            try:
                val = float(cleaned)
            except ValueError as err:
                raise ValueError(
                    f"TotalCharges string '{v}' must be numeric or whitespace."
                ) from err
            if val < 0:
                raise ValueError("TotalCharges cannot be negative.")
            return val
        raise ValueError(f"Unsupported type for TotalCharges: {type(v).__name__}")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }
    )


class ChurnPredictionResponse(BaseModel):
    """Response model containing binary churn prediction and probability score."""

    prediction: Literal["Churn", "No Churn"] = Field(
        ...,
        description="Predicted churn category ('Churn' or 'No Churn').",
        examples=["Churn"],
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability of customer churn (between 0.0 and 1.0).",
        examples=[0.82],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": "Churn",
                "probability": 0.82,
            }
        }
    )


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(
        default="healthy",
        description="Operational status of the service.",
        examples=["healthy"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
            }
        }
    )
