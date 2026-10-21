from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Transaction(BaseModel):
    """Raw transaction fields, named as in the IEEE-CIS CSVs.

    The named fields are the ones the feature pipeline relies on. Any other raw column
    (V1..V339, C1..C14, D2..D15, M1..M9, id_01..id_38) may be sent as an extra field;
    anything not sent is treated as missing, which LightGBM handles natively.
    """

    model_config = ConfigDict(extra="allow")

    TransactionID: int | None = None
    TransactionDT: int = Field(..., description="seconds since the dataset's reference point")
    TransactionAmt: float = Field(..., gt=0)
    ProductCD: str
    card1: int
    card2: float | None = None
    card3: float | None = None
    card4: str | None = None
    card5: float | None = None
    card6: str | None = None
    addr1: float | None = None
    addr2: float | None = None
    dist1: float | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None
    D1: float | None = None
    DeviceType: str | None = None
    DeviceInfo: str | None = None
    id_30: str | None = None
    id_31: str | None = None


class CostOverride(BaseModel):
    chargeback_fee: float | None = None
    friction_cost: float | None = None
    lost_margin_rate: float | None = None
    review_cost: float | None = None
    reviewer_accuracy: float | None = Field(None, ge=0.0, le=1.0)


class ScoreRequest(BaseModel):
    transaction: Transaction
    costs: CostOverride | None = None


class Reason(BaseModel):
    feature: str
    contribution: float
    text: str


class ScoreResponse(BaseModel):
    transaction_id: int
    p_raw: float
    p_calibrated: float
    decision: str
    expected_cost: dict[str, float]
    break_even_probability: float
    reason_codes: list[Reason]
    model_version: str
