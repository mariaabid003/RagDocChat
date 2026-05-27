"""
Run this script to verify your MongoDB Atlas connection is working.
Usage:  python test_mongo.py
"""
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / ".env")

MONGODB_URI = os.getenv("MONGODB_URI", "")

async def test():
    if not MONGODB_URI or MONGODB_URI == "your_mongodb_atlas_uri_here":
        print("[ERROR] MONGODB_URI is not set in backend/.env")
        print("    Open backend/.env and replace 'your_mongodb_atlas_uri_here'")
        print("    with your actual Atlas connection string.")
        return

    try:
        import motor.motor_asyncio
        client = motor.motor_asyncio.AsyncIOMotorClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=6000,
        )
        await client.admin.command("ping")
        db = client["dochat"]
        # List existing collections
        cols = await db.list_collection_names()
        print("[OK] MongoDB Atlas connected successfully!")
        print(f"    Database : dochat")
        print(f"    Collections: {cols if cols else '(empty -- will be created on first use)'}")
        client.close()
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        print()
        print("Common fixes:")
        print("  1. Check your username/password in the connection string")
        print("  2. In Atlas → Network Access → add your IP (or 0.0.0.0/0 for all)")
        print("  3. Make sure the cluster is not paused")

asyncio.run(test())
