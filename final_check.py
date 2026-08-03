import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def test_connection():
    driver = GraphDatabase.driver(URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            session.run("MERGE (c:Document {title: 'Constitution of India'})")
            print("✅ Successfully connected and created the root node!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
    finally:
        driver.close()


if __name__ == "__main__":
    test_connection()