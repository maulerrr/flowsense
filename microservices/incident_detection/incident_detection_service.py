from kafka import KafkaConsumer, KafkaProducer
import json

# Initialize Kafka Consumer and Producer
consumer = KafkaConsumer(
    'anomaly_data',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Simulated incident classification (simple rule-based logic)
def classify_incidents():
    for message in consumer:
        anomaly_data = message.value
        if anomaly_data['anomaly']:
            print(f"Incident detected: {anomaly_data['vehicle_id']}")
            producer.send('incident_data', {'vehicle_id': anomaly_data['vehicle_id'], 'incident': 'Accident'})
        else:
            producer.send('incident_data', {'vehicle_id': anomaly_data['vehicle_id'], 'incident': 'No incident'})

if __name__ == "__main__":
    classify_incidents()
