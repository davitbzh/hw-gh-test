set -e

echo "$PWD"

echo "New data ingestion pipeline starts"
python3 pipeline.py
