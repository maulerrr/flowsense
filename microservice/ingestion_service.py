#!/usr/bin/env python3
"""
Flask API: POST /ingest → publish to 'raw-events'.
"""

from flask import Flask, request, jsonify
from kafka import KafkaProducer
import json, os

app = Flask(__name__)
BROKER = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")

producer = KafkaProducer(
    bootstrap_servers=BROKER,
    value_serializer=lambda v: json.dumps(v).encode()
)

@app.route("/ingest", methods=["POST"])
def ingest():
    evt = request.json
    producer.send("raw-events", evt)
    return jsonify(status="queued", event=evt), 202

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
