import time
import random

# Simulated traffic data
data = [
    {'vehicle_id': 'V1', 'speed': 70, 'location': 'X1'},
    {'vehicle_id': 'V2', 'speed': 40, 'location': 'X2'},
    {'vehicle_id': 'V3', 'speed': 20, 'location': 'X3'},  # This is an anomaly
    {'vehicle_id': 'V4', 'speed': 80, 'location': 'X4'}
]

def detect_anomalies(record):
    speed = record['speed']
    if speed < 30:
        return True
    return False

def classify_incidents(record, anomaly):
    if anomaly:
        return 'Accident'
    return 'No incident'

def process_traffic_data():
    for record in data:
        print(f"Processing data: {record}")
        anomaly = detect_anomalies(record)
        incident = classify_incidents(record, anomaly)
        if incident == 'Accident':
            print(f"Incident detected: {incident} for vehicle {record['vehicle_id']}")
        else:
            print(f"No incident for vehicle {record['vehicle_id']}")
        time.sleep(1)

if __name__ == "__main__":
    process_traffic_data()
    