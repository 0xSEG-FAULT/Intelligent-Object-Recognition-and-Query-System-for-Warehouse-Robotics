import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class RoboticsRAG:
    def __init__(self, docs_dir="data/docs"):
        print("Initializing RAG System...")
        
        # 1. Load & Split Docs
        os.makedirs(docs_dir, exist_ok=True)
        if len(os.listdir(docs_dir)) == 0:
            self._create_dummy_docs(docs_dir)
            
        loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents(docs)
        
        # 2. Vector Store (Embeddings)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_db = FAISS.from_documents(splits, self.embeddings)
        
        # 3. Local LLM (TinyLlama - 1.1B params)
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
           
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

    def query(self, user_query, detected_object=None):
        """
        Retrieves context and answers the query.
        """
        # Augment query with context
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
        docs = {
            "fragile_handling.txt": """
Fragile objects such as glassware, ceramics, and sensitive electronics must be handled using soft-grip end effectors.
Maximum acceleration should not exceed 0.5 m/s².
The robotic arm must reduce speed near placement zones.
Avoid sudden rotational motion.
Use vibration dampening when available.
""",
            "hazardous_materials.txt": """
Hazardous materials require insulated grippers.
Maximum tilt angle must not exceed 15 degrees.
Transport speed must remain below 0.4 m/s.
Emergency stop must be triggered if leakage is detected.
Maintain safe distance during handling.
""",
            "heavy_object_guidelines.txt": """
Objects over 20 kg require hydraulic lift assist.
Verify load capacity before lifting.
Operate in low-speed torque mode.
Ensure path is clear before movement.
Do not exceed joint torque limits.
""",
            "conveyor_safety.txt": """
Maintain 30 cm clearance from moving conveyor belts.
Use synchronized motion mode.
Do not retrieve items during high-speed operation.
Inspect sensors daily.
""",
            "battery_safety.txt": """
Lithium batteries must not be punctured or overheated.
Store between 15°C and 25°C.
Isolate damaged batteries immediately.
Use fire-resistant containers if required.
""",
            "robot_arm_specifications.txt": """
Maximum payload: 25 kg.
Maximum reach: 1.2 meters.
Operating voltage: 48V DC.
Recommended temperature: 5°C to 40°C.
Weekly calibration required.
""",
            "sensor_troubleshooting.txt": """
Clean camera lens if detection fails.
Recalibrate ultrasonic sensors.
Check lighting conditions.
Restart perception module if latency increases.
""",
            "motor_overheat_protocol.txt": """
If motor temperature exceeds 75°C, reduce load.
Activate cooling fans.
Pause operations for 5 minutes.
Resume only below 50°C.
""",
            "emergency_shutdown.txt": """
Press emergency stop in case of fire or malfunction.
Cut main power within 10 seconds.
Notify maintenance supervisor.
Do not restart without inspection.
""",
            "gripper_calibration.txt": """
Calibrate gripper pressure weekly.
Standard force range: 10N to 60N.
Test using calibration block.
Replace worn grip pads immediately.
"""
        }

        for filename, content in docs.items():
            with open(os.path.join(dir_path, filename), "w") as f:
                f.write(content.strip())
