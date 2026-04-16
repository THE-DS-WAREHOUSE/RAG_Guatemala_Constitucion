from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
#%%
load_dotenv()
#%%
class LegalRAGSystem:
    def __init__(self):
        self.db_directory = "./chroma_db"

        print("Loading embedding model...")
        # MUST use the exact same embedding model you used to build the DB
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        print("Connecting to Vector Database...")
        self.vector_db = Chroma(
            persist_directory=self.db_directory,
            embedding_function=self.embeddings
        )

        # We turn the database into a "Retriever" that fetches the top 3 matches
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})

        print("Connecting to OpenAI (ChatGPT)...")
        # Using gpt-3.5-turbo (or gpt-4o-mini) is cheap, fast, and perfect for this.
        # temperature=0 makes the AI highly factual and less "creative" (important for law)
        self.llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    def build_qa_chain(self):
        """Builds the prompt and connects the Retriever to the LLM."""

        # The System Prompt is crucial. It forces the AI to ONLY use your documents.
        system_prompt = (
            "Eres un asistente legal experto en la Constitución de Guatemala. "
            "Usa los siguientes fragmentos de contexto recuperados para responder a la pregunta. "
            "Si no sabes la respuesta o no está en el contexto, simplemente di que no lo sabes "
            "basado en la información proporcionada. No inventes respuestas.\n\n"
            "Contexto:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Create a chain that stuffs the retrieved documents into the prompt
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)

        # Combine the retriever and the QA chain into the final RAG pipeline
        rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)

        return rag_chain

#%%
if __name__ == "__main__":
    rag_system = LegalRAGSystem()
    qa_chain = rag_system.build_qa_chain()

    print("\n" + "=" * 50)
    print("SISTEMA RAG LEGAL INICIADO (Escribe 'salir' para terminar)")
    print("=" * 50)

    while True:
        user_query = input("\nPregunta: ")

        if user_query.lower() in ['salir', 'exit', 'quit']:
            print("Cerrando el sistema. ¡Hasta luego!")
            break

        if not user_query.strip():
            continue

        print("Pensando...\n")

        # Execute the RAG pipeline
        response = qa_chain.invoke({"input": user_query})

        print(f"Respuesta de ChatGPT:\n{response['answer']}")

        # Print the sources so you can verify the AI isn't hallucinating
        print("\n--- Fuentes utilizadas ---")
        for i, doc in enumerate(response['context']):
            source = doc.metadata.get('source_file', 'Desconocido')
            # Just print the first 100 characters of the source to verify
            print(f"{i + 1}. {source}: {doc.page_content[:100]}...")