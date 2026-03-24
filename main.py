import os
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from langchain.chains import RetrievalQA

from langchain.llms.base import LLM

# Optional Claude API
import anthropic

# Load API key
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

# -------- Step 1: Load document --------
loader = PyPDFLoader("docs/sample.pdf")
documents = loader.load()

# -------- Step 2: Split text --------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# -------- Step 3: Create embeddings --------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------- Step 4: Store in vector DB --------
db = Chroma.from_documents(
    docs,
    embedding_model,
    persist_directory="vector_db"
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# -------- Step 5: Claude LLM wrapper --------
class ClaudeLLM(LLM):

    def _call(self, prompt, stop=None):
        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text

    @property
    def _llm_type(self):
        return "claude"

llm = ClaudeLLM()

# -------- Step 6: Retrieval QA chain --------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# -------- Step 7: Chat loop --------
print("📄 Document AI Ready! Ask questions from document.\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    result = qa_chain(query)

    print("\n🤖 Answer:")
    print(result["result"])

    print("\n📚 Source:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source"))
