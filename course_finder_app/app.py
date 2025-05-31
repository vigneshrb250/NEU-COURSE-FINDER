import os
from flask import Flask, request, render_template, redirect, url_for, session
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from dotenv import load_dotenv

# Load Hugging Face token from .env if present
load_dotenv()
os.environ["HUGGINGFACE_API_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

# Set up Chroma DB
# db = chromadb.PersistentClient(path=r"C:\Users\vigne\Desktop\Higher studies\Northeastern University Boston\Courses\Semester 2\DS5983(LLMs)\NEU-COURSE-FINDER\NeuCourses_Chroma_db")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "NeuCourses_Chroma_db")
db = chromadb.PersistentClient(path=DB_PATH)
chroma_collection = db.get_or_create_collection("NeuCourses_Chroma_db")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Load index from stored vector DB
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

# Use Zephyr-7B from HuggingFace Inference API
llm = HuggingFaceInferenceAPI(model_name="HuggingFaceH4/zephyr-7b-beta")
query_engine = index.as_query_engine(llm=llm, response_mode="tree_summarize")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_query = request.form["user_query"]
        response = query_engine.query(user_query)

        # Save to session temporarily
        session["response_text"] = str(response)
        session["source_courses"] = [
    (
        node.node.metadata.get("title", "Unknown Course"),
        node.node.text  
    )
    for node in response.source_nodes
]

        return redirect(url_for("home"))  # Redirect after POST

    # GET request: retrieve and clear session data
    response_text = session.pop("response_text", None)
    source_courses = session.pop("source_courses", [])

    return render_template("index.html", response=response_text, sources=source_courses)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

