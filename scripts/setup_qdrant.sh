#!/bin/bash
echo "Starting Qdrant via Docker..."
docker compose up -d
echo "Qdrant is running on http://localhost:6333"
