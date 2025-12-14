from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_community.vectorstores import FAISS

from operator import itemgetter
import os

# -- Set up vector store path --
VECTOR_STORE_PATH = "./faiss_db"

# -- RAGPipeline class --
class RAGPipeline:
    # Constructor for RAGPipeline
    def __init__(self):
        self.embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

        self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", 
                temperature=0.1, 
                max_output_tokens=1024
            )

        # Load vector store
        if os.path.exists(VECTOR_STORE_PATH):
            self.vector_store = FAISS.load_local(
                VECTOR_STORE_PATH, 
                self.embeddings_model, 
                allow_dangerous_deserialization=True
            )
        else:
            self.vector_store = None

        self._build_prompt()
        self._build_chain()

    # -- PDF Loader --
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        loader = PyMuPDFLoader(pdf_path)
        return loader.load() # returns a list of documents
    
    # -- Build a vector store from the extracted text --
    def build_vector_store(self, text:str):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )

        chunks = splitter.split_documents(text)

        new_store = FAISS.from_documents(chunks, embedding=self.embeddings_model)

        # Merge with existing vector store if it exists
        if self.vector_store:
            self.vector_store.merge_from(new_store)
        else:
            self.vector_store = new_store

        # Save the vector store to the local directory
        self.vector_store.save_local(VECTOR_STORE_PATH)

        # Rebuild chain with vector store
        self._build_chain()

        return len(chunks)
    
    # -- Prompt Template --
    def _build_prompt(self):
        self.prompt = ChatPromptTemplate.from_messages([
           (
                "system",
                "You are a helpful AI assistant. "
                "Answer ONLY using the provided context. "
                "If the answer is not found, say 'I don't know'."
            ),

            (
                'user', 
                'Context: {context}\n\nQuestion: {question}\n\nAnswer in a concise manner.'
            )
        ]
        )

    # -- RAG Chain --
    def _build_chain(self, k: int = 3):
        if self.vector_store is None:
            return

        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        self.rag_chain = (
            # -- Retrive documents --
            # RunnableParallel is used to run the question and docs in parallel -> it returns a dictionary with the question and docs
            RunnableParallel(
                question=itemgetter("question"),          # get the question (input)
                docs = itemgetter("question") | retriever # returns a list of documents
            )

            # -- Build prompt inputs --
            # RunnableLambda is used to format the documents to a string -> transforms the documents to a string
            | {
                'question': itemgetter("question"), 
                'context': itemgetter("docs") | RunnableLambda(format_docs), # Format the documents to a string
                'docs' : itemgetter("docs")                                            # get the documents (output)
            }

            # -- Generate answer and keep sources --
            | RunnableParallel(
                answer = self.prompt | self.llm | StrOutputParser(),
                docs = itemgetter("docs")
            )
        )
    
    # -- Query the vector store to get an answer --
    def query(self, query: str, k: int = 3):
        if not self.vector_store:
            raise ValueError("Vector store is not built. Please build it before querying.")
        
        self._build_chain(k=k)

        # invoke chain
        result = self.rag_chain.invoke({"question": query})

        return {
            "answer": result["answer"],
            "sources": [
                {
                    "page": doc.metadata.get("page"),
                    "content": doc.page_content[:100]
                }
                for doc in result["docs"]
            ]
        }