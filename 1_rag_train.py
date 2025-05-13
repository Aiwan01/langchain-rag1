from dotenv import load_dotenv
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "story1.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")


if not os.path.exists(persistent_directory):
    print("Persistence directory not exist, Initializing vector store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} not exist. Please check")

    #now loading file
    loader = TextLoader(file_path)
    documents = loader.load()

    # converting text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)


    #displaying splitted text
    print("\n Document chunk data")
    print(f"Number document chunk {len(docs)}")
    print(f"Sample chunk: \n{docs[0].page_content}")


    #creating embedding
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("\n Embedding of text finished.")


    # storing data in chroma db
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("FInished creating  vector store")

else:
    print("vector store already exist, No need ReInitialized")


print("Storing data and vector completed")