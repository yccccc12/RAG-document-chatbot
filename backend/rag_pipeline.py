from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_community.vectorstores import FAISS
from langchain.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder, PromptTemplate
from tavily import TavilyClient
from operator import itemgetter
import os

# -- Set up vector store path --
VECTOR_STORE_PATH = "./faiss_db"

# -- RAGPipeline class --
class RAGPipeline:
    # Constructor for RAGPipeline
    def __init__(self, google_api_key: str, tavily_api_key: str):
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key=google_api_key)

        self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite", 
                temperature=0.1, 
                max_output_tokens=1024,
                google_api_key=google_api_key
            )

        self.tavily_client = TavilyClient(api_key=tavily_api_key)

        # Load vector store
        if os.path.exists(VECTOR_STORE_PATH):
            self.vector_store = FAISS.load_local(
                VECTOR_STORE_PATH, 
                self.embeddings_model, 
                allow_dangerous_deserialization=True
            )
        else:
            self.vector_store = None

        self.chat_history = []

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
    
    # -- Prompt Template for main RAG Agent --
    def _build_prompt(self):
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful research assistant. "
                "1. First, check the Chat History. If the user is asking about themselves (like their name), answer from memory. "
                "2. If the question is technical, use the provided PDF or Web Context. "
                "3. If the Context is empty or irrelevant to a personal question, rely on your conversation history."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                'user', 
                'Context from Search/PDF: {context}\n\nQuestion: {question}'
            )
        ])

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
                'docs' : itemgetter("docs")                                  # get the documents (output)
            }

            # -- Generate answer and keep sources --
            | RunnableParallel(
                answer = self.prompt | self.llm | StrOutputParser(),
                docs = itemgetter("docs")
            )
        )
    
    # -- Use LLM as a judge to check if the context is relevant to the question asked --
    def _check_relevance(self, query: str, context: str) -> bool:
        prompt = PromptTemplate.from_template(
            """
            You are a judge assessing whether the provided context is relevant to answer the question.
            Question: {query}
            Context: {context}

            Answer with a "Yes" if the context is relevant, otherwise answer "No".
            Strictly respond with only "Yes" or "No" only.
            """
        )

        grader = prompt | self.llm | StrOutputParser()
        response = grader.invoke({"query": query, "context": context}).lower().strip()

        return "yes" == response

    # -- Prompt refiner to rephrase follow-up questions as standalone questions --
    def _refine_prompt(self, query: str) -> str:
        prompt = PromptTemplate.from_template(
            """
            You are a prompt refiner. Given the conversation history and a follow-up question,
            rephrase the follow-up question to be a standalone question.

            Note:
            1. Do not add any additional information.
            2. Do not answer the question.
            3. Your task is only to rephrase the question.
            4. If the question is already standalone, return it as is.

            Conversation History: {chat_history}
            Follow-Up Question: {question}
            """
        )
        refiner = prompt | self.llm | StrOutputParser()
        return refiner.invoke({"chat_history": self.chat_history, "question": query})
    
    def stream_query(self, query: str):
        # Refine the prompt if there is chat history
        if self.chat_history:
            query = self._refine_prompt(query)
        
        # Retrieve from Vector Store
        docs = []
        context = ""
        sources = []
        is_relevant = False

        if self.vector_store is not None:
            docs = self.vector_store.as_retriever().invoke(query)
            context = "\n\n".join([d.page_content for d in docs])

            # Check if retrieved context is relevant to the query
            is_relevant = self._check_relevance(query, context)

        if not is_relevant:
            search_results = self.tavily_client.search(query=query, max_results=3)

            # Format Web Sources
            for res in search_results['results']:
                sources.append({
                    "type": "web",
                    "url": res.get("url"),
                    "content": res.get("content")
                })

            # Rewrite context to include search results
            context = "\n\nWeb Search Results:\n"
            for idx, res in enumerate(search_results['results'], 1):
                context += f"{idx}. {res.get('content')}\n Source: {res.get('url')}\n\n"
            
        else:
            # Extract PDF Sources
            for doc in docs:
                sources.append({
                    "type": "pdf",
                    "page": doc.metadata.get("page", 0) + 1,
                    "content": doc.page_content
                })

        # Chain execution with Memory
        chain = self.prompt | self.llm | StrOutputParser()
        
        full_response = ""
        for chunk in chain.stream({ "question": query, "context": context, "chat_history": self.chat_history }):
            full_response += chunk
            yield {
                "type": "answer", 
                "content": chunk
                }

        # Save to Memory
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=full_response))
        
        # Keep last 10 messages
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]

        # Emit sources at the end
        yield {
            "type": "sources",
            "content": sources
        }

def get_pipeline(google_key: str, tavily_key: str) -> RAGPipeline:
    return RAGPipeline(google_api_key=google_key, tavily_api_key=tavily_key)