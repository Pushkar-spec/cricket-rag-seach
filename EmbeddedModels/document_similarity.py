from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader("EmbeddedModels/cricket.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    provider="auto"   # or pin explicitly if this model needs a specific provider
)
model = ChatHuggingFace(llm=llm)

while True:
    query = input("\nEnter your query (or type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    results = vectorstore.similarity_search(query, k=2)

    retrieved_text = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""Answer the question based only on the context below.

Context:
{retrieved_text}

Question: {query}

Answer:"""

    response = model.invoke(prompt)
    print("\nAnswer:", response.content)