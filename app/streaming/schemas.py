from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class StreamTransaction(BaseModel):
    user_id: str
    amount: float

    transaction_type: str
    currency: str = "USD"

    destination_id: Optional[str] = None
    ip_address: Optional[str] = None

    created_at: datetime
