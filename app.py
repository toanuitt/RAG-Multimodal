import streamlit as st
from extractfile import process_pdf
from embedding import database 
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import AzureChatOpenAI, AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from config import api_key, azure_endpoint, api_version

# Azure OpenAI Embeddings setup
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="embedding",
    model="text-embedding-3-large",  # Change to your preferred embedding model
    openai_api_version=api_version,
    api_key=api_key,
    azure_endpoint=azure_endpoint,
)

# ChromaDB setup with the specified embedding function
db = Chroma(
    persist_directory="chroma_db_store",  # Path to your ChromaDB directory
    embedding_function=embeddings  # Use Azure OpenAI embeddings
)

# Azure Chat OpenAI setup for generating responses
llm = AzureChatOpenAI(
    deployment_name="gpt4o-mini",  # Replace with your GPT-4 deployment name
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
    max_tokens=4000,
    temperature=0.1
)

# Define your custom prompt template
custom_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# Create a question generator chain
question_generator = LLMChain(llm=llm, prompt=CUSTOM_QUESTION_PROMPT)

# Create the retrieval chain
retriever = db.as_retriever()
combine_docs_chain = create_stuff_documents_chain(llm, PromptTemplate.from_template("{context}\n\nQuestion: {question}\n\nAnswer:"))

def format_chat_history(chat_history):
    return "\n".join(f"Human: {human}\nAI: {ai}" for human, ai in chat_history)

def conversational_retrieval_chain(input_dict):
    chat_history = input_dict.get("chat_history", [])
    question = input_dict["question"]
    
    # Generate a standalone question
    standalone_question = question_generator.run({"chat_history": format_chat_history(chat_history), "question": question})
    
    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(standalone_question)
    
    # Ensure docs are in the correct format
    formatted_docs = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in docs]
    
    # Generate the final answer
    answer = combine_docs_chain.invoke({
        "context": formatted_docs,
        "question": standalone_question
    })
    
    return {"answer": answer, "source_documents": docs}

# Create the final chain
chain = RunnablePassthrough() | conversational_retrieval_chain

# Streamlit UI
st.title("AI Chatbot with File Upload")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Process button
if uploaded_file and st.button("Process"):
    # Save the uploaded file locally
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    # Process the PDF file to extract text and images
    process_pdf("uploaded.pdf")
    
    # Save the extracted text and images to ChromaDB
    database("D:\RAG\documents")  # Replace with the actual path where `process_pdf` saves content
    
    st.success(f"File '{uploaded_file.name}' processed and added to the database!")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    chat_history = [(m["content"], st.session_state.messages[i+1]["content"]) 
                    for i, m in enumerate(st.session_state.messages[:-1]) if m["role"] == "user"]
    response, sources = chain.invoke({"question": prompt, "chat_history": chat_history}).values()

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
