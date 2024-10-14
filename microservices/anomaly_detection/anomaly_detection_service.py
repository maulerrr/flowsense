from kafka import KafkaConsumer, KafkaProducer
import json

# Initialize Kafka Consumer and Producer
consumer = KafkaConsumer(
    'traffic_data',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Anomaly detection based on speed (simple rule: speed < 30 is an anomaly)
def detect_anomalies():
    for message in consumer:
        record = message.value
        speed = record['speed']
        if speed < 30:
            print(f"Anomaly detected: {record}")
            producer.send('anomaly_data', {'vehicle_id': record['vehicle_id'], 'anomaly': True})
        else:
            producer.send('anomaly_data', {'vehicle_id': record['vehicle_id'], 'anomaly': False})

if __name__ == "__main__":
    detect_anomalies()
