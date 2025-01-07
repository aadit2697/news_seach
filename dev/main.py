import os
import streamlit as st
import pickle
import time
import langchain
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
#from langchain.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
#from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

#load_dotenv()

st.title("News search tool")

st.sidebar.header("News Article URLs")


urls= []
for i  in range(3):
    url= st.sidebar.text_input(f"URL:  {i+1}")
    urls.append(url)
process_url_clicked = st.sidebar.button("Process URLs")

file_path= "faiss_store_openai.pkl"

#progress bar

main_placeholder= st.empty()

if process_url_clicked:
    loader= UnstructuredURLLoader(urls= urls)
    main_placeholder.text("Data loading started...✅✅✅")
    data= loader.load()
    #split data
    text_splitter= RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','],
        chunk_size=1000) 
     
    docs= text_splitter.split_documents(data)
    # create embeddings
    embeddings= OpenAIEmbeddings(model = "text-embedding-3-large")
    vectorstore_openai= FAISS.from_documents(docs,embedding= embeddings)
    main_placeholder.text("Embedding vector started building...✅✅✅")
     # Save the FAISS index to a pickle file
    vectorstore_openai.save_local(file_path)

llm = OpenAI(temperature=0.9, max_tokens=500)
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # Use the same model as during saving
        vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)

        # Create the chain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

        # Run the query
        result = chain({"question": query}, return_only_outputs=True)

        # Display the result
        st.header("Answer")
        st.write(result["answer"])

        # display source if available

        sources = result.get("sources","") # we use get instead of calling like this result["answer"] because sources might not always be present

        if sources:
            st.subheader("Sources: ")
            sources_list =  sources.split('\n') 
            for source in sources_list:
                st.write(source)