#requirements
!pip install flask
!pip install qdrant-client
!pip install sentence-transformers==2.2.2
!pip install huggingface_hub==0.14.1
!pip install PyPDF2
!pip install python-docx
!pip install pandas
!pip install google.generativeai
!pip install langgraph
!pip install Flask-CORS
!pip install flask_pymongo
!pip install langchain
!pip install pytesseract 
!pip install pillow
!pip install deep-translator



#app.py
from flask import Flask, jsonify
from flask_cors import CORS
from routes.upload import upload_bp
from routes.query import query_bp
from routes.documents import documents_bp
from routes.user_routes import user_bp
from routes.section_routes import section_bp
from routes.quiz_routes import quiz_bp
from routes.activity_routes import activity_bp
from routes.student_stats_routes import student_stats_bp
from routes.translate import translate_bp
from routes.suggest import suggest_bp   
from routes.tts import tts_bp
from routes.quiz_attempt_routes import quiz_attempt_bp
from routes.auth_routes import auth_bp # Ensure this import is present

app = Flask(__name__)

# Configure CORS properly
CORS(app, origins=["http://localhost:3000"], supports_credentials=True)

# Register all blueprints with /api prefix
app.register_blueprint(upload_bp, url_prefix='/api')
app.register_blueprint(query_bp, url_prefix='/api')
app.register_blueprint(documents_bp, url_prefix='/api')
app.register_blueprint(translate_bp, url_prefix='/api')
app.register_blueprint(tts_bp, url_prefix='/api')
app.register_blueprint(suggest_bp, url_prefix='/api')
app.register_blueprint(user_bp, url_prefix='/api')
app.register_blueprint(section_bp, url_prefix='/api')
app.register_blueprint(quiz_bp, url_prefix='/api')
app.register_blueprint(activity_bp, url_prefix='/api')
app.register_blueprint(student_stats_bp, url_prefix='/api')
app.register_blueprint(quiz_attempt_bp, url_prefix='/api')
app.register_blueprint(auth_bp, url_prefix='/api') # Ensure this registration is present

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Agentic RAG API is running. Use /api endpoints."})

@app.route("/api", methods=["GET"])
def api_home():
    return jsonify({"message": "API endpoints available"})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)



#authdecorator
from functools import wraps
from flask import request, jsonify
from db.mongo import mongo_db
from bson.objectid import ObjectId

def role_required(roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get user email from headers (sent by frontend)
            user_email = request.headers.get('X-User-Email')
            
            if not user_email:
                return jsonify({'message': 'User email header is missing'}), 401

            # Find user in database
            user = mongo_db.users.find_one({"email": user_email})
            if not user:
                return jsonify({'message': 'User not found'}), 401

            # Check if user role is allowed
            if user['role'] not in roles:
                return jsonify({'message': 'Unauthorized'}), 403

            # Create current_user object
            current_user = {
                'id': str(user['_id']),
                'email': user['email'],
                'name': user.get('name', ''),
                'role': user['role']
            }

            return f(current_user, *args, **kwargs)
        return decorated_function
    return decorator



#initdb
from config import MONGO_CONNECTION_STRING, CA_FILE
from pymongo import MongoClient
from datetime import datetime
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash

# Connect to MongoDB
try:
    client = MongoClient(MONGO_CONNECTION_STRING, tlsCAFile=CA_FILE)
    db = client["AgenticRag"]  # Your database name
    print("Connected to MongoDB successfully!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()

# Define collections
users_collection = db["users"]
sections_collection = db["sections"]
quizzes_collection = db["quizzes"]
chat_collection = db["chat_history"]
documents_collection = db["documents"]
activity_submissions_collection = db["activity_submissions"] # Added for completeness
quiz_attempts_collection = db["quiz_attempts"] # Added for completeness
publish_schedules_collection = db["publish_schedules"] # Added for completeness


def seed_data():
    print("Seeding initial data...")
    # Clear existing data
    users_collection.drop()
    sections_collection.drop()
    quizzes_collection.drop()
    chat_collection.drop()
    documents_collection.drop()
    activity_submissions_collection.drop() # Drop new collections too
    quiz_attempts_collection.drop()
    publish_schedules_collection.drop()
    print("Dropped existing collections.")

    # Seed Users
    admin_obj_id = ObjectId()
    teacher_obj_id = ObjectId()
    student_obj_id = ObjectId()

    users_data = [
        {"_id": admin_obj_id, "email": "admin@example.com", "name": "Admin User", "role": "admin", "password": generate_password_hash("admin123")},
        {"_id": teacher_obj_id, "email": "teacher@example.com", "name": "Teacher Jane", "role": "teacher", "password": generate_password_hash("teacher123")},
        {"_id": student_obj_id, "email": "student@example.com", "name": "Student John", "role": "student", "password": generate_password_hash("student123")},
    ]
    users_collection.insert_many(users_data)
    print(f"Seeded {len(users_data)} users.")

    # Seed Sections
    section1_obj_id = ObjectId()
    section2_obj_id = ObjectId()
    sections_data = [
        {"_id": section1_obj_id, "name": "Introduction to AI", "description": "Basics of Artificial Intelligence.", "teacher_id": teacher_obj_id, "created_at": datetime.utcnow(), "updated_at": datetime.utcnow()},
        {"_id": section2_obj_id, "name": "Advanced Machine Learning", "description": "Deep dive into ML algorithms.", "teacher_id": teacher_obj_id, "created_at": datetime.utcnow(), "updated_at": datetime.utcnow()},
    ]
    sections_collection.insert_many(sections_data)
    print(f"Seeded {len(sections_data)} sections.")

    # Seed Quizzes
    quiz1_obj_id = ObjectId()
    quiz2_obj_id = ObjectId()
    quizzes_data = [
        {
            "_id": quiz1_obj_id,
            "title": "AI Fundamentals Quiz",
            "description": "Quiz on basic AI concepts.",
            "section_id": section1_obj_id, # Use ObjectId
            "created_by": teacher_obj_id, # Use ObjectId
            "is_enabled": True, # Set to True for students to see it
            "questions": [
                {"question_text": "What does AI stand for?", "options": ["Artificial Intelligence", "Automated Information", "Advanced Integration", "Algorithmic Innovation"], "correct_answer": "Artificial Intelligence"},
                {"question_text": "Which of these is a type of AI?", "options": ["Machine Learning", "Data Storage", "Cloud Computing", "Network Security"], "correct_answer": "Machine Learning"},
                {"question_text": "What is the goal of Artificial General Intelligence (AGI)?", "options": ["To perform specific tasks efficiently", "To mimic human-like intelligence across various tasks", "To process large datasets", "To automate repetitive jobs"], "correct_answer": "To mimic human-like intelligence across various tasks"},
            ],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "_id": quiz2_obj_id,
            "title": "ML Algorithms Quiz",
            "description": "Quiz on common machine learning algorithms.",
            "section_id": section2_obj_id, # Use ObjectId
            "created_by": teacher_obj_id, # Use ObjectId
            "is_enabled": False, # Keep as False for testing publishing
            "questions": [
                {"question_text": "What is a common supervised learning algorithm?", "options": ["K-Means", "Linear Regression", "PCA", "Apriori"], "correct_answer": "Linear Regression"},
                {"question_text": "Which algorithm is used for clustering?", "options": ["Decision Tree", "K-Means", "Support Vector Machine", "Naive Bayes"], "correct_answer": "K-Means"},
                {"question_text": "What does ' overfitting' mean in machine learning?", "options": ["Model performs well on training data but poorly on new data", "Model performs poorly on training data", "Model is too simple", "Model is too fast"], "correct_answer": "Model performs well on training data but poorly on new data"},
            ],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        },
        {
            "_id": ObjectId(), # New quiz without section_id
            "title": "General Knowledge Quiz",
            "description": "A quiz on various general topics.",
            "section_id": None, # Explicitly set to None
            "created_by": teacher_obj_id,
            "is_enabled": True,
            "questions": [
                {"question_text": "What is the capital of France?", "options": ["Berlin", "Madrid", "Paris", "Rome"], "correct_answer": "Paris"},
                {"question_text": "Which planet is known as the Red Planet?", "options": ["Earth", "Mars", "Jupiter", "Venus"], "correct_answer": "Mars"},
            ],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
    ]
    quizzes_collection.insert_many(quizzes_data)
    print(f"Seeded {len(quizzes_data)} quizzes.")
    print("Data seeding complete!")

if __name__ == "__main__":
    seed_data()
    client.close()
    print("MongoDB connection closed.")



#createtestusers
from werkzeug.security import generate_password_hash
from db.mongo import mongo_db
from datetime import datetime

def create_test_users():
    # Create admin user
    admin_user = {
        "name": "Admin User",
        "email": "admin@example.com",
        "password": generate_password_hash("admin123"),
        "role": "admin",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    # Create teacher user
    teacher_user = {
        "name": "Teacher User",
        "email": "teacher@example.com", 
        "password": generate_password_hash("teacher123"),
        "role": "teacher",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    # Create student user
    student_user = {
        "name": "Student User",
        "email": "student@example.com",
        "password": generate_password_hash("student123"),
        "role": "student", 
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    # Check if users already exist
    if not mongo_db.users.find_one({"email": "admin@example.com"}):
        mongo_db.users.insert_one(admin_user)
        print("Admin user created: admin@example.com / admin123")
    
    if not mongo_db.users.find_one({"email": "teacher@example.com"}):
        mongo_db.users.insert_one(teacher_user)
        print("Teacher user created: teacher@example.com / teacher123")
        
    if not mongo_db.users.find_one({"email": "student@example.com"}):
        mongo_db.users.insert_one(student_user)
        print("Student user created: student@example.com / student123")

if __name__ == "__main__":
    create_test_users()



#createsections

from db.mongo import mongo_db
from bson.objectid import ObjectId
from datetime import datetime

def create_default_sections():
    # First, get teacher users
    teachers = list(mongo_db.users.find({"role": "teacher"}))
    
    if not teachers:
        print("No teachers found. Please create teacher users first.")
        return
    
    # Create default sections
    default_sections = [
        {"name": "Mathematics", "teacher_id": teachers[0]["_id"]},
        {"name": "Science", "teacher_id": teachers[0]["_id"]},
        {"name": "English", "teacher_id": teachers[0]["_id"]},
    ]
    
    for section_data in default_sections:
        # Check if section already exists
        existing_section = mongo_db.sections.find_one({"name": section_data["name"]})
        if not existing_section:
            section_data.update({
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            result = mongo_db.sections.insert_one(section_data)
            print(f"Created section: {section_data['name']} with ID: {result.inserted_id}")
        else:
            print(f"Section {section_data['name']} already exists")

if __name__ == "__main__":
    create_default_sections()



#userroutes
from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
from models.types import doc_to_dict
from auth_decorators import role_required
from db.mongo import mongo_db
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash
from datetime import datetime

user_bp = Blueprint("user", __name__)

@user_bp.route("/users", methods=["GET"])
@cross_origin(origins=["http://localhost:3000"])
@role_required(["admin"])
def get_users(current_user):
    users = []
    for user_doc in mongo_db.users.find():
        user_dict = doc_to_dict(user_doc)
        # Remove password from response
        user_dict.pop('password', None)
        users.append(user_dict)
    return jsonify(users)

@user_bp.route("/users", methods=["POST"])
@cross_origin(origins=["http://localhost:3000"])
@role_required(["admin"])
def create_user(current_user):
    data = request.get_json()
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    role = data.get("role")

    if not all([name, email, password, role]):
        return jsonify({"error": "Bad Request", "message": "All fields are required"}), 400

    if role not in ["admin", "teacher", "student"]:
        return jsonify({"error": "Bad Request", "message": "Invalid role"}), 400

    # Check if user already exists
    existing_user = mongo_db.users.find_one({"email": email})
    if existing_user:
        return jsonify({"error": "Conflict", "message": "User with this email already exists"}), 409

    new_user_data = {
        "name": name,
        "email": email,
        "password": generate_password_hash(password),
        "role": role,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    result = mongo_db.users.insert_one(new_user_data)
    new_user_data['id'] = str(result.inserted_id)
    new_user_data.pop('password', None)  # Remove password from response
    
    return jsonify(new_user_data), 201

@user_bp.route("/users/<user_id>", methods=["PUT"])
@cross_origin(origins=["http://localhost:3000"])
@role_required(["admin"])
def update_user(current_user, user_id):
    try:
        user_obj_id = ObjectId(user_id)
    except Exception:
        return jsonify({"error": "Bad Request", "message": "Invalid User ID format"}), 400

    user_doc = mongo_db.users.find_one({"_id": user_obj_id})
    if not user_doc:
        return jsonify({"error": "Not Found", "message": "User not found"}), 404

    data = request.get_json()
    update_data = {
        "name": data.get("name", user_doc["name"]),
        "email": data.get("email", user_doc["email"]),
        "role": data.get("role", user_doc["role"]),
        "updated_at": datetime.utcnow()
    }

    # Update password if provided
    if data.get("password"):
        update_data["password"] = generate_password_hash(data["password"])

    mongo_db.users.update_one({"_id": user_obj_id}, {"$set": update_data})
    updated_user_doc = mongo_db.users.find_one({"_id": user_obj_id})
    updated_user_dict = doc_to_dict(updated_user_doc)
    updated_user_dict.pop('password', None)  # Remove password from response
    
    return jsonify(updated_user_dict)

@user_bp.route("/users/<user_id>", methods=["DELETE"])
@cross_origin(origins=["http://localhost:3000"])
@role_required(["admin"])
def delete_user(current_user, user_id):
    try:
        user_obj_id = ObjectId(user_id)
    except Exception:
        return jsonify({"error": "Bad Request", "message": "Invalid User ID format"}), 400

    user_doc = mongo_db.users.find_one({"_id": user_obj_id})
    if not user_doc:
        return jsonify({"error": "Not Found", "message": "User not found"}), 404

    mongo_db.users.delete_one({"_id": user_obj_id})
    return jsonify({"message": "User deleted successfully"}), 200



from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
from rag.extract_text import extract_text
from rag.embedding import embed_and_store, embed_image_and_store
import os

upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/upload', methods=['POST'])
@cross_origin(origin="http://localhost:3000")
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    total_chunks = 0
    for file in files:
        try:
            filename = file.filename
            extension = os.path.splitext(filename)[-1].lower()

            # Use in-memory file object (no need to save locally)
            file_stream = file.stream

            # Extract text directly from file stream
            content = extract_text(file_stream, filename)

            # Route image vs non-image embeddings
            if extension in [".png", ".jpg", ".jpeg"]:
                chunk_count = embed_image_and_store(filename, content)
            else:
                chunk_count = embed_and_store(filename, content)

            total_chunks += chunk_count

        except Exception as e:
            print(f"Error processing {file.filename}: {str(e)}")
            return jsonify({"error": f"{file.filename} failed: {str(e)}"}), 500

    return jsonify({"message": f"{total_chunks} chunks embedded from {len(files)} file(s)"}), 200



#elevenlabs
from flask import Blueprint, request, jsonify, Response
from flask_cors import cross_origin
import requests
import os

tts_bp = Blueprint("tts", __name__)

# ElevenLabs Text-to-Speech Route
@tts_bp.route("/text-to-speech", methods=["POST", "OPTIONS"])
@cross_origin(origins=["http://localhost:3000"])
def elevenlabs_tts():  # ✅ RENAMED to avoid conflict
    data = request.get_json()
    text = data.get("text", "")
    voice_id = data.get("voice_id", "CxUF1MnX2dESXqaELxCQ")  # Default voice

    if not text:
        return jsonify({"error": "No text provided"}), 400

    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    if not ELEVENLABS_API_KEY:
        return jsonify({"error": "ElevenLabs API key not configured"}), 500

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }

    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            return Response(
                response.content,
                mimetype="audio/mpeg",
                headers={
                    "Content-Disposition": "attachment; filename=speech.mp3",
                    "Access-Control-Allow-Origin": "http://localhost:3000",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type"
                }
            )
        else:
            error_msg = f"ElevenLabs API error: {response.status_code}"
            if response.text:
                error_msg += f" - {response.text}"
            return jsonify({"error": error_msg}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Fallback: Google TTS
@tts_bp.route("/text-to-speech-simple", methods=["POST", "OPTIONS"])
@cross_origin(origins=["http://localhost:3000"])
def gtts_fallback():  # ✅ RENAMED to avoid conflict
    """Fallback TTS using Google Text-to-Speech (gTTS)"""
    try:
        from gtts import gTTS
        import io

        data = request.get_json()
        text = data.get("text", "")
        lang = data.get("lang", "en")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        tts = gTTS(text=text, lang=lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return Response(
            audio_buffer.getvalue(),
            mimetype="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=speech.mp3",
                "Access-Control-Allow-Origin": "http://localhost:3000",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )

    except ImportError:
        return jsonify({"error": "gTTS not installed. Install with: pip install gtts"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500



#embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient
import uuid
from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME
from sentence_transformers import SentenceTransformer

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def embed_and_store(filename, content):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(content)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=EMBED_MODEL.encode(chunk).tolist(),
            payload={"source": filename, "text": chunk}
        )
        for chunk in chunks
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(chunks)
def embed_image_and_store(filename, ocr_text):
    if not ocr_text.strip():
        return 0

    embedding = EMBED_MODEL.encode(ocr_text).tolist()
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={
            "source": filename,
            "type": "image",
            "image_path": f"static/uploads/{filename}",
            "ocr_text": ocr_text
        }
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[point])
    return 1


#extract-text
import os
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # change path if different


def extract_text(file, filename):
    ext = os.path.splitext(filename)[-1].lower()
    if ext == '.pdf':
        reader = PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif ext == '.docx':
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif ext == '.txt':
        return file.read().decode("utf-8")
    elif ext == '.csv':
        df = pd.read_csv(file)
        return df.to_string(index=False)
    elif ext in ['.png', '.jpg', '.jpeg']:
        try:
            image = Image.open(file)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            raise ValueError(f"Image OCR failed: {str(e)}")

    
    else:
        raise ValueError(f"Unsupported file type: {ext}")



#graph
from langgraph.graph.state import StateGraph, START, END
from models.types import ChatState
from rag.nodes import (
    extract_context_node, classify_agent_node, 
    build_response_node, update_history_node, route_agent
)

graph_builder = StateGraph(ChatState)
graph_builder.add_node("extract_context", extract_context_node)
graph_builder.add_node("classify_agent", classify_agent_node)
graph_builder.add_node("technical_agent", build_response_node("technical"))
graph_builder.add_node("customer_agent", build_response_node("customer"))
graph_builder.add_node("common_agent", build_response_node("general"))
graph_builder.add_node("update_history", update_history_node)

graph_builder.add_edge(START, "extract_context")
graph_builder.add_edge("extract_context", "classify_agent")
graph_builder.add_conditional_edges("classify_agent", route_agent)
graph_builder.add_edge("technical_agent", "update_history")
graph_builder.add_edge("customer_agent", "update_history")
graph_builder.add_edge("common_agent", "update_history")
graph_builder.add_edge("update_history", END)

graph = graph_builder.compile()

from models.types import ChatState, ChatMessage # Import ChatState and ChatMessage
# Assuming 'graph' is an object or function that needs to be defined here
# For demonstration, let's define a placeholder run_graph function
# You will replace this with your actual RAG graph implementation.

def run_graph(chat_state: ChatState):
    """
    Placeholder for your RAG graph logic.
    Processes the chat_state and returns a result.
    """
    query = chat_state['query']
    chat_history = chat_state['chat_history'] # This will now be List[ChatMessage]
    user_email = chat_state['user_email']
    selected_file = chat_state['selected_file']

    # Example: Simple echo or mock RAG response
    print(f"Processing query: {query} for user: {user_email}")
    print(f"Chat history length: {len(chat_history)}")
    
    # In a real RAG system, you would:
    # 1. Retrieve context based on query and selected_file
    # 2. Use an LLM to generate an answer based on query, context, and chat_history
    # 3. Update chat_state with the answer and context_chunks

    mock_answer = f"Hello {user_email}, I received your query: '{query}'. This is a mock response from the RAG graph."
    mock_context_chunks = ["Context chunk 1 from file.", "Context chunk 2 related to query."]

    # Update the chat_state with the answer and context chunks
    chat_state['answer'] = mock_answer
    chat_state['context_chunks'] = mock_context_chunks

    return {
        "answer": chat_state['answer'],
        "context_chunks": chat_state['context_chunks'],
        "chat_history": [msg.dict() for msg in chat_state['chat_history']] # Convert ChatMessage back to dict for jsonify
    }

# If your graph is an object, you might have something like:
# class RAGGraph:
#     def __init__(self):
#         # Initialize RAG components
#         pass
#     def run(self, chat_state: ChatState):
#         # RAG logic here
#         return {"answer": "...", "context_chunks": [...]}
# graph = RAGGraph() # Instantiate your graph object if needed



#nodes
from config import GEMINI_API_KEY, COLLECTION_NAME
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import google.generativeai as genai
from rag.utils import clean_markdown
from db.mongo import chat_collection

genai.configure(api_key=GEMINI_API_KEY)

client = QdrantClient(url="https://2ed85abb-e606-4167-8d2e-ce4185f33997.us-east4-0.gcp.cloud.qdrant.io", 
                      api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.UYr-iYmbfZzhyr-lGQBlMlMuYQIAxriQhZd6af7vLq4")

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def extract_context_node(state: dict) -> dict:
    query_vector = EMBED_MODEL.encode(state["query"]).tolist()

    filter_condition = None
    if state.get("selected_file"):
        filter_condition = {
            "must": [
                {
                    "key": "source",
                    "match": {"value": state["selected_file"]}
                }
            ]
        }

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=3,
        with_payload=True,
        query_filter=filter_condition
    )

    return {
        "context_chunks": [res.payload["text"] for res in results]
    }


def classify_agent_node(state: dict) -> dict:
    context = "\n".join(state.get("context_chunks", []))
    prompt = f"""
You are a routing agent responsible for determining the appropriate agent to handle the user query and how the selected agent should respond.

Agent types: technical, customer, common

Rules:
- technical: engineering-related queries
- customer: support/product/service queries
- common: general queries

Return only one: technical, customer, or common.

Query: {state['query']}

Context:
{context}
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return {"agent_type": response.text.strip().lower()}


def build_response_node(role: str):
    def node(state: dict) -> dict:
        context = "\n".join(state.get("context_chunks", []))
        prompt = f"""
You are a helpful {role} agent. Respond only using the context.

Ensure formatting:
- No **bold**, *italic*, or bullet points
- Use paragraphs only

Query:
{state['query']}

Context:
{context}
"""
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        cleaned_response = clean_markdown(response.text.strip())
        return {"answer": cleaned_response}
    return node


def update_history_node(state: dict) -> dict:
    history = state.get("chat_history", [])
    chat_entry = {
        "question": state["query"],
        "answer": state.get("answer", ""),
        "selected_file": state.get("selected_file", ""),
        "agent": state.get("agent_type", "common"),
        "user_email": state.get("user_email", "anonymous"),
    }

    try:
        inserted = chat_collection.insert_one(chat_entry)
        chat_entry["id"] = str(inserted.inserted_id)
    except Exception as e:
        print(f"[MongoDB Error] Failed to insert chat: {e}")

    history.append(chat_entry)
    return {"chat_history": history}


def route_agent(state: dict) -> str:
    return {
        "technical": "technical_agent",
        "customer": "customer_agent",
    }.get(state.get("agent_type", ""), "common_agent")



#utils
import re

def clean_markdown(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"\n?\d+\.\s*", "\n", text)
    text = re.sub(r"\n?[-•]\s*", "\n", text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()

def get_unique_filenames(documents):
    seen = set()
    unique_files = []
    for doc in documents:
        fname = doc["filename"]
        if fname not in seen:
            seen.add(fname)
            unique_files.append(fname)
    return unique_files


