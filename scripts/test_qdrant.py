from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "test_collection"


def test_qdrant_connection():
    print("\n--- Testing Qdrant connection ---")
    try:
        client = QdrantClient(url=QDRANT_URL)

        # Create a small test collection (4-dimensional dummy vectors)
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created.")

        # Insert a dummy point
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=1,
                    vector=[0.1, 0.2, 0.3, 0.4],
                    payload={"doc_id": "DOC001", "text": "test chunk from Indian Contract Act"},
                )
            ],
        )
        print("Test vector inserted.")

        # Search for it
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=[0.1, 0.2, 0.3, 0.4],
            limit=1,
        )
        print("Search successful. Result:")
        for point in results.points:
            print(f"- id={point.id}, score={point.score}, payload={point.payload}")

        print("\n✅ Qdrant connection fully verified.")

    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")


if __name__ == "__main__":
    test_qdrant_connection()