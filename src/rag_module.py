import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class RoboticsRAG:
    def __init__(self, docs_dir="data/docs"):
        print("Initializing RAG System...")
        
        # 1. Load & Split Docs [cite: 58]
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir, exist_ok=True)
            self._create_dummy_docs(docs_dir) # Helper to create sample docs if empty
            
        loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents(docs)
        
        # 2. Vector Store (Embeddings)
        # Using a small, fast embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_db = FAISS.from_documents(splits, self.embeddings)
        
        # 3. Local LLM (TinyLlama - 1.1B params, fast on CPU/GPU) [cite: 60]
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=200,
            temperature=0.7
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

    def query(self, user_query, detected_object=None):
        """
        Retrieves context and answers the query.
        """
        # Augment query with context [cite: 59]
        context_prefix = ""
        if detected_object:
            context_prefix = f"Regarding the '{detected_object}' item: "
        
        # Retrieve relevant docs
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke(user_query)
        context_text = "\n".join([d.page_content for d in docs])
        
        # Prompt Engineering
        prompt = f"""<|system|>
You are a warehouse robotics expert. Use the context below to answer the user's question briefly.
Context:
{context_text}
</s>
<|user|>
{context_prefix}{user_query}
</s>
<|assistant|>"""
        
        return self.llm.invoke(prompt)

    def _create_dummy_docs(self, dir_path):
        # Creates sample docs if none exist [cite: 51]
        content = """
        [Protocol: Handling Fragile Items]
        Fragile items (glass, ceramics) must be handled with the soft-grip effector.
        Maximum acceleration: 0.5 m/s^2.
        
        [Protocol: Heavy Lifting]
        Heavy items (>20kg) require the hydraulic lift assist.
        Ensure the path is clear before engaging.
        
        [Protocol: Hazardous Materials]
        Hazardous items (batteries, chemicals) trigger a red light warning.
        Do not tilt > 15 degrees.
        """
        with open(os.path.join(dir_path, "protocols.txt"), "w") as f:
            f.write(content)