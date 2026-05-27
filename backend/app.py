import os
import uuid
import shutil
import warnings
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from contextlib import asynccontextmanager

# Suppress passlib/bcrypt version-mismatch warning (passlib 1.7.4 + bcrypt ≥4)
warnings.filterwarnings("ignore", ".*error reading bcrypt version.*")
warnings.filterwarnings("ignore", ".*trapped.*error reading bcrypt.*")

from dotenv import load_dotenv

# Always load .env relative to this file so it works regardless of cwd
_HERE = Path(__file__).parent
load_dotenv(_HERE / ".env")

# ── FastAPI ────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# ── Auth ───────────────────────────────────────────────────────────────────────
from jose import JWTError, jwt
from passlib.context import CryptContext

# ── MongoDB ────────────────────────────────────────────────────────────────────
import motor.motor_asyncio
from bson import ObjectId

# ── Document loaders ───────────────────────────────────────────────────────────
import fitz  # PyMuPDF
import docx as docxlib

# ── Embeddings & vector store ──────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
import chromadb

# ── LLM ───────────────────────────────────────────────────────────────────────
import anthropic

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MONGODB_URI       = os.getenv("MONGODB_URI", "")
JWT_SECRET        = os.getenv("JWT_SECRET", "change_this_super_secret_key_in_production")
JWT_ALGORITHM     = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_DAYS   = int(os.getenv("JWT_EXPIRE_DAYS", "7"))

# Use absolute paths so the server works from any working directory
UPLOAD_DIR = _HERE / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR = _HERE / "chroma_db"
CHROMA_DIR.mkdir(exist_ok=True)

# ── Password hashing ───────────────────────────────────────────────────────────
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ── HTTP Bearer ────────────────────────────────────────────────────────────────
bearer_scheme = HTTPBearer()

# ── Embedding model ────────────────────────────────────────────────────────────
print("[INFO] Loading embedding model (all-MiniLM-L6-v2)…")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("[OK]  Embedding model ready.")

# ── ChromaDB ───────────────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)

# ── MongoDB ────────────────────────────────────────────────────────────────────
mongo_client = None
db = None

# ── Anthropic async client (created once at startup) ──────────────────────────
anthropic_client: Optional[anthropic.AsyncAnthropic] = None

# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN (replaces deprecated @app.on_event)
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown resources."""
    global mongo_client, db, anthropic_client

    # ── Anthropic async client ─────────────────────────────────────────────────
    if ANTHROPIC_API_KEY:
        anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        print("[OK]  Anthropic client ready.")
    else:
        print("[WARN] ANTHROPIC_API_KEY not set — /chat will be unavailable")

    # ── MongoDB ────────────────────────────────────────────────────────────────
    if not MONGODB_URI:
        print("[WARN] MONGODB_URI not set — auth/persistence disabled")
    else:
        try:
            mongo_client = motor.motor_asyncio.AsyncIOMotorClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=8000,
            )
            await mongo_client.admin.command("ping")
            db = mongo_client["dochat"]

            # Create indexes — idempotent, safe to call every restart
            await db.users.create_index("email", unique=True)
            await db.documents.create_index([("user_id", 1), ("doc_id", 1)])
            await db.chats.create_index([("user_id", 1), ("timestamp", -1)])
            await db.chats.create_index("session_id")
            print("[OK]  MongoDB Atlas connected successfully (database: dochat)")
        except Exception as exc:
            print(f"[ERROR] MongoDB connection failed: {exc}")
            print("        Check MONGODB_URI in .env and Atlas Network Access (allow 0.0.0.0/0).")
            mongo_client = None
            db = None

    yield  # ── application runs here ──────────────────────────────────────────

    # ── Shutdown ───────────────────────────────────────────────────────────────
    if mongo_client:
        mongo_client.close()
    if anthropic_client:
        try:
            await anthropic_client.close()
        except Exception:
            pass  # older SDK versions may not expose close()


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="DocChat AI API", version="2.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # JWT is sent as a header — no credential cookies
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def hash_password(plain: str) -> str:
    return pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


def create_access_token(user_id: str, email: str, name: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "exp": expire,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        if not user_id:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token payload")
        return {
            "user_id": user_id,
            "email": payload.get("email", ""),
            "name": payload.get("name", ""),
        }
    except JWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token invalid or expired")


def require_db():
    if db is None:
        raise HTTPException(
            503,
            "Database not available — check MONGODB_URI in .env and server logs",
        )


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """Split text into overlapping word-based chunks."""
    if chunk_size <= 0:
        chunk_size = 500
    overlap = min(overlap, chunk_size - 1)  # prevent infinite loop
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + chunk_size]))
        i += chunk_size - overlap
    return chunks


def extract_text(file_path: Path, filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        doc = fitz.open(str(file_path))
        return "\n".join(page.get_text() for page in doc)
    elif ext == "txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")
    elif ext == "docx":
        doc = docxlib.Document(str(file_path))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError(f"Unsupported file type: .{ext}")


def embed(texts: List[str]) -> List[List[float]]:
    return embedder.encode(texts, show_progress_bar=False).tolist()


def get_user_chunks(user_id: str):
    """
    Return (chunk_ids, metadatas_by_chunk_id) for a given user from ChromaDB.
    """
    result = collection.get(include=["metadatas"])
    ids       = result["ids"]
    metadatas = result["metadatas"]

    user_chunk_ids = []
    user_metadatas = {}
    for cid, meta in zip(ids, metadatas):
        if meta.get("user_id") == user_id:
            user_chunk_ids.append(cid)
            user_metadatas[cid] = meta
    return user_chunk_ids, user_metadatas


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    session_id: str
    session_title: str


# ══════════════════════════════════════════════════════════════════════════════
# ROOT / HEALTH
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "status": "DocChat AI API v2.2 running",
        "db_connected": db is not None,
        "ai_ready": anthropic_client is not None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/auth/register")
async def register(req: RegisterRequest):
    require_db()

    name  = req.name.strip()
    email = req.email.lower().strip()

    if not name:
        raise HTTPException(400, "Name cannot be empty")
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")

    existing = await db.users.find_one({"email": email})
    if existing:
        raise HTTPException(400, "An account with this email already exists")

    # ── MongoDB schema: users ──────────────────────────────────────────────────
    # { name, email, password_hash, created_at }
    user_doc = {
        "name":          name,
        "email":         email,
        "password_hash": hash_password(req.password),
        "created_at":    datetime.now(timezone.utc),
    }
    result  = await db.users.insert_one(user_doc)
    user_id = str(result.inserted_id)
    token   = create_access_token(user_id, email, name)

    return {
        "access_token": token,
        "token_type":   "bearer",
        "name":         name,
        "email":        email,
    }


@app.post("/auth/login")
async def login(req: LoginRequest):
    require_db()

    email = req.email.lower().strip()
    user  = await db.users.find_one({"email": email})

    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password")

    user_id = str(user["_id"])
    token   = create_access_token(user_id, user["email"], user["name"])

    return {
        "access_token": token,
        "token_type":   "bearer",
        "name":         user["name"],
        "email":        user["email"],
    }


@app.get("/auth/me")
async def me(current_user: dict = Depends(get_current_user)):
    return current_user


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT ROUTES (protected)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    allowed = {"pdf", "txt", "docx"}
    ext = file.filename.lower().rsplit(".", 1)[-1] if "." in file.filename else ""
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type '.{ext}'. Allowed: PDF, TXT, DOCX")

    doc_id    = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"

    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        text = extract_text(save_path, file.filename)
    except Exception as exc:
        save_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Failed to extract text: {exc}")

    if not text.strip():
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, "Document appears empty or unreadable")

    chunks     = chunk_text(text)
    chunk_ids  = [f"{doc_id}_{i}" for i in range(len(chunks))]
    embeddings = embed(chunks)

    # ChromaDB stores: embedding vectors + chunk text + metadata
    metadatas = [
        {
            "doc_id":      doc_id,
            "filename":    file.filename,
            "chunk_index": i,
            "user_id":     current_user["user_id"],  # stored as string in ChromaDB
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    # ── MongoDB schema: documents ──────────────────────────────────────────────
    # { filename, upload_date, user_id, doc_id, chunk_count }
    if db is not None:
        await db.documents.insert_one({
            "doc_id":      doc_id,
            "filename":    file.filename,
            "chunk_count": len(chunks),
            "upload_date": datetime.now(timezone.utc),
            "user_id":     ObjectId(current_user["user_id"]),
        })

    return {
        "success":     True,
        "doc_id":      doc_id,
        "filename":    file.filename,
        "chunk_count": len(chunks),
    }


@app.get("/documents")
async def list_documents(current_user: dict = Depends(get_current_user)):
    """List documents for the current user."""
    # Prefer MongoDB (faster, authoritative) when available
    if db is not None:
        cursor = db.documents.find(
            {"user_id": ObjectId(current_user["user_id"])},
            {"_id": 0, "doc_id": 1, "filename": 1, "chunk_count": 1, "upload_date": 1},
        ).sort("upload_date", -1)
        docs = []
        async for doc in cursor:
            docs.append({
                "doc_id":      doc["doc_id"],
                "filename":    doc["filename"],
                "chunk_count": doc["chunk_count"],
            })
        return {"documents": docs}

    # Fallback: derive from ChromaDB
    _, user_metadatas = get_user_chunks(current_user["user_id"])
    seen: dict = {}
    for meta in user_metadatas.values():
        did = meta["doc_id"]
        if did not in seen:
            seen[did] = {"doc_id": did, "filename": meta["filename"], "chunk_count": 0}
        seen[did]["chunk_count"] += 1
    return {"documents": list(seen.values())}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, current_user: dict = Depends(get_current_user)):
    results = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
    if not results["ids"]:
        raise HTTPException(404, "Document not found")

    # Verify ownership
    owner = results["metadatas"][0].get("user_id")
    if owner != current_user["user_id"]:
        raise HTTPException(403, "You do not own this document")

    collection.delete(ids=results["ids"])

    for f in UPLOAD_DIR.glob(f"{doc_id}_*"):
        f.unlink(missing_ok=True)

    if db is not None:
        await db.documents.delete_one(
            {"doc_id": doc_id, "user_id": ObjectId(current_user["user_id"])}
        )

    return {"success": True, "deleted_chunks": len(results["ids"])}


@app.delete("/documents")
async def clear_all_documents(current_user: dict = Depends(get_current_user)):
    """Clear ALL documents belonging to current user."""
    user_chunk_ids, user_metadatas = get_user_chunks(current_user["user_id"])

    if user_chunk_ids:
        collection.delete(ids=user_chunk_ids)

    # Remove uploaded files
    doc_ids = {m["doc_id"] for m in user_metadatas.values()}
    for did in doc_ids:
        for f in UPLOAD_DIR.glob(f"{did}_*"):
            f.unlink(missing_ok=True)

    if db is not None:
        await db.documents.delete_many({"user_id": ObjectId(current_user["user_id"])})

    return {"success": True, "message": "All your documents have been cleared"}


# ══════════════════════════════════════════════════════════════════════════════
# CHAT ROUTES (protected)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/chat")
async def chat(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    if not ANTHROPIC_API_KEY or anthropic_client is None:
        raise HTTPException(500, "ANTHROPIC_API_KEY not configured — check .env")

    # Get this user's chunk IDs from ChromaDB
    user_chunk_ids, _ = get_user_chunks(current_user["user_id"])

    if not user_chunk_ids:
        raise HTTPException(
            400,
            "No documents indexed yet — upload a document first using the Docs tab"
        )

    query_embedding = embed([req.question])[0]

    top_k     = max(1, req.top_k or 5)
    n_results = min(top_k, len(user_chunk_ids))

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"user_id": current_user["user_id"]},
        include=["documents", "metadatas", "distances"],
    )

    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not chunks:
        raise HTTPException(400, "No relevant content found in your documents")

    context_parts, sources = [], []
    for i, (chunk, meta, dist) in enumerate(zip(chunks, metadatas, distances)):
        context_parts.append(f"[Source {i + 1}: {meta['filename']}]\n{chunk}")
        sources.append({
            "filename":  meta["filename"],
            "doc_id":    meta["doc_id"],
            "relevance": round(1 - dist, 4),
        })

    context = "\n\n---\n\n".join(context_parts)
    system_prompt = (
        "You are a helpful AI assistant that answers questions based strictly on the "
        "provided document context. If the answer cannot be found in the context, "
        "say so honestly. Always cite which source document your answer comes from. "
        "Format your response clearly using markdown when appropriate."
    )
    user_message = (
        f"Context from uploaded documents:\n\n{context}\n\n---\n\nQuestion: {req.question}"
    )

    message = await anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    answer = message.content[0].text

    # ── MongoDB schema: chats ──────────────────────────────────────────────────
    # { question, answer, document_id, timestamp, session_id, session_title, user_id, sources }
    if db is not None:
        # Resolve primary document_id as a MongoDB ObjectId reference
        primary_doc_id  = sources[0]["doc_id"] if sources else None
        mongo_doc       = None
        if primary_doc_id:
            mongo_doc = await db.documents.find_one({"doc_id": primary_doc_id})

        await db.chats.insert_one({
            "session_id":    req.session_id,
            "session_title": req.session_title,
            "question":      req.question,
            "answer":        answer,
            "document_id":   mongo_doc["_id"] if mongo_doc else None,   # ObjectId ref
            "user_id":       ObjectId(current_user["user_id"]),
            "sources":       sources,
            "timestamp":     datetime.now(timezone.utc),
        })

    return {"answer": answer, "sources": sources, "chunks_used": len(chunks)}


@app.get("/chats")
async def get_chats(
    current_user: dict = Depends(get_current_user),
    limit: int = 100,
):
    """Return chat messages grouped by session for the current user."""
    if db is None:
        return {"sessions": []}

    cursor = db.chats.find(
        {"user_id": ObjectId(current_user["user_id"])},
    ).sort("timestamp", 1)   # ascending → messages in correct order

    sessions_map: dict = {}
    async for doc in cursor:
        sid = doc.get("session_id", str(doc["_id"]))
        if sid not in sessions_map:
            sessions_map[sid] = {
                "id":        sid,
                "title":     doc.get("session_title", "Chat"),
                "messages":  [],
                "createdAt": int(doc["timestamp"].timestamp() * 1000),
                "msgCount":  0,
            }

        sessions_map[sid]["messages"].append({
            "role":    "user",
            "content": doc["question"],
            "sources": [],
        })
        sessions_map[sid]["messages"].append({
            "role":    "assistant",
            "content": doc["answer"],
            "sources": doc.get("sources", []),
        })
        sessions_map[sid]["msgCount"] += 1

    sessions = list(sessions_map.values())
    # Sort sessions newest-first (for sidebar display)
    sessions.sort(key=lambda x: x["createdAt"], reverse=True)

    return {"sessions": sessions[:limit]}


@app.delete("/chats/{session_id}")
async def delete_chat_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete an entire chat session."""
    if db is None:
        raise HTTPException(503, "Database not available")

    result = await db.chats.delete_many({
        "session_id": session_id,
        "user_id":    ObjectId(current_user["user_id"]),
    })

    return {"success": True, "deleted_count": result.deleted_count}