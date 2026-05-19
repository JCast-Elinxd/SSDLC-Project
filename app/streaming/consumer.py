from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    "transactions",
    bootstrap_servers="kafka:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="earliest",
    group_id="transactions-group"
)

print("Consumer started...")

for message in consumer:
    tx = message.value

    print("EVENT RECEIVED:")
    print(tx)