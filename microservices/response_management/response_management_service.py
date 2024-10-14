from kafka import KafkaConsumer
import json

# Initialize Kafka Consumer
consumer = KafkaConsumer(
    'incident_data',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Simulated response management
def manage_response():
    for message in consumer:
        incident = message.value
        if incident['incident'] == 'Accident':
            print(f"Alert: Accident detected for vehicle {incident['vehicle_id']}. Sending response.")
        else:
            print(f"No incident for vehicle {incident['vehicle_id']}.")

if __name__ == "__main__":
    manage_response()
