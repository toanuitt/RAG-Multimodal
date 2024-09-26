from summery import *
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from config import api_key, azure_endpoint, api_version
def process_all_files(folder_path):
    # Loop through all files in the folder
    content = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            # Process each file based on its extension
            if filename.endswith(".png"):
                print(f"Processing image: {filename}")
                content.append(summary_img(file_path))  # Assuming process_pdf handles PDFs            
            elif filename.endswith(".txt"):
                print(f"Processing txt file:{filename}")
                content.append(summary_text(file_path))
            else:
                print(f"Skipping unsupported file format: {filename}")
    return content

def database(path):
    # Initialize embeddings using AzureOpenAIEmbeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="embedding",
        model="text-embedding-3-large",  # Change to your preferred embedding model
        openai_api_version=api_version,
        api_key=api_key,
        azure_endpoint=azure_endpoint
    )

    # Specify a directory to persist the Chroma database
    persist_directory = "chroma_db_store"  # Directory to save the database

    # Ensure the directory exists
    os.makedirs(persist_directory, exist_ok=True)

    # Initialize Chroma with persistence enabled
    db = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    # Process your files and store in the database
    raw_documents = process_all_files(path)  # Assuming this function loads and preprocesses your docs

    for text in raw_documents:
        # Split each text into chunks of 1000 characters with no overlap
        chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_text(text)

        # Convert chunks to Documents format
        documents = [Document(page_content=chunk) for chunk in chunks]

        # Add documents to the Chroma database
        db.add_documents(documents)

    return db





