
from kafka import KafkaConsumer, KafkaProducer
import json


consumer = KafkaConsumer(
    'traffic_data',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)


def predict_incidents():
    for message in consumer:
        record = message.value
        
        if record['speed'] < 20:
            producer.send('prediction_data', {'vehicle_id': record['vehicle_id'], 'prediction': 'Potential incident'})
        else:
            producer.send('prediction_data', {'vehicle_id': record['vehicle_id'], 'prediction': 'No issue'})

if __name__ == "__main__":
    predict_incidents()
