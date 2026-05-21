import os
import json

from kafka import KafkaProducer


KAFKA_SERVER = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka:9092"
)

TOPIC = "transactions"


producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)


def publish_transaction(data: dict):
    producer.send(TOPIC, value=data)

    # SOLO para desarrollo
    producer.flush()