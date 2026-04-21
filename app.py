from flask import Flask, render_template, request, jsonify
from src.helper import generate_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

app = Flask(__name__)

load_dotenv()

os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

embeddings = generate_embeddings()

index_name = "medical-chatbot"

# Initialize Vector Store
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Initialize Retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize local LLM (MedGemma)
llm = ChatOllama(model="medgemma1.5:4b-it-bf16")

# Set up the Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the RAG Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(f"User Input: {input}")
    
    # Process through RAG chain
    response = rag_chain.invoke({"input": input})
    print(f"Response: {response['answer']}")
    
    return jsonify({"answer": response["answer"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)