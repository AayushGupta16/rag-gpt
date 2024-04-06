import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
import re
import nest_asyncio
OPENAI_API_KEY: os.getenv("OPENAI_API_TOKEN")
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # Load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # Store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # Load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Settings for LlamaIndex
Settings.chunk_size = 512
Settings.chunk_overlap = 64
Settings.llm = OpenAI(model="gpt-4-0125-preview")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Setup Query Engine Tool for the index
query_engine_tool = QueryEngineTool(
    query_engine=index.as_query_engine(),
    metadata=ToolMetadata(
        name="vector_index",
        description="Provides information about documents text. "
            "Use a detailed plain text question as input to the tool.",    
    ),
)

# Create an OpenAIAgent with the query engine tool
agent = OpenAIAgent.from_tools(
    tools=[query_engine_tool],
    llm_kwargs={
        "temperature": 0,
        "max_tokens": 500,
        "model_name": "gpt-4-0125-preview",
        "openai_api_key": os.getenv("OPENAI_API_TOKEN"),
    },
    verbose=True  # Enables verbose output to see the agent's internal process
)



# Chatbot loop
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    # Use the processed input for the chatbot response
    prompt_template = (
        "Instructions- You are to assist with querring the information from the provided documents and then accurately represent and present that information."
        "There are provided tools which you can use to access document specifics"
        "\nUser:"
    )
    response = agent.chat(prompt_template + user_input)
    print(f"Chatbot: {response}")





def process_message(user_input):
    prompt_template = (
        "Instructions- You are to assist with querring the information from the provided documents and then accurately represent and present that information."
        "There are provided tools which you can use to access document specifics"
        "\nUser:"
    )
    return str(agent.chat(prompt_template + user_input))