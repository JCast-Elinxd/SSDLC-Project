from fastapi import APIRouter

from app.streaming.schemas import StreamTransaction
from app.streaming.producer import publish_transaction


router = APIRouter(
    prefix="/stream",
    tags=["streaming"]
)


@router.post("/transactions")
def stream_transaction(tx: StreamTransaction):

    publish_transaction(
        tx.model_dump(mode="json")
    )

    return {
        "status": "queued",
        "message": "Transaction published to Kafka"
    }