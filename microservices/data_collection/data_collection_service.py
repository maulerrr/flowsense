from kafka import KafkaProducer
import json
import time

# Sample traffic data
data = [
    {'vehicle_id': 'V1', 'speed': 70, 'location': 'X1', 'timestamp': '2024-10-10 10:00:00'},
    {'vehicle_id': 'V2', 'speed': 40, 'location': 'X2', 'timestamp': '2024-10-10 10:01:00'},
    # Add more synthetic data as needed
]

# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Simulate data stream
def send_traffic_data():
    for record in data:
        producer.send('traffic_data', record)
        print(f"Sent data: {record}")
        time.sleep(1)

if __name__ == "__main__":
    send_traffic_data()
