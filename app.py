import os
import openai
import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from streamlit_extras.add_vertical_space import add_vertical_space
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
import fitz
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd

openai.api_type = "azure"
openai.api_base = "https://rfq-abb.openai.azure.com/"
openai.api_key = "192d27f4fc584d76abd8a5eb978dcedf"
openai.api_version = "2023-03-15-preview"

with st.sidebar:
    st.title("Ask the Doc")
    st.markdown("This is a LLM powered chatbot that can answer your questions about RFQs")
    add_vertical_space(5)
    st.write("Made by")
    image = Image.open('logo.png')
    st.sidebar.image(image, width=200)


def main():
    import time
    start_time = time.time()
    llm = AzureChatOpenAI(temperature=0,deployment_name="rfq", openai_api_key=openai.api_key, openai_api_base=openai.api_base, openai_api_version=openai.api_version)
    embeddings = OpenAIEmbeddings(deployment_id="rfq-embeddings", chunk_size=1,openai_api_key=openai.api_key, openai_api_base=openai.api_base, openai_api_version=openai.api_version)

    st.header("RFQ Extraction")
    pdf=st.file_uploader("Upload your RFQ", type=["pdf"])
    

    if pdf is not None:
        file_path = os.path.join( pdf.name)
        with open(file_path,"wb") as f: 
            f.write(pdf.getbuffer())         
            st.success("Saved File")
        pdf_file=fitz.open(file_path)

        # st.write(pdf.name)
        text=""
        for page in pdf_file:
            text+=page.get_text()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_text(text)
        # st.write(text)
        # store_name=pdf.name[:-4]
        # if os.path.exists(f"{store_name}.pkl"):
        #     with open(f"{store_name}.pkl","rb") as f:
        #         vectorstore = pickle.load(f)
        #     st.write("Embedding Loaded")
        # else:
        directory="vector_store"
        vectorstore = FAISS.from_texts(splits, embedding=embeddings)
        vectorstore.save_local(directory)
        vector_index = FAISS.load_local('vector_store', 
                                        OpenAIEmbeddings(openai_api_key = openai.api_key,
                                                        deployment_id = "rfq-embeddings"))
        retriever = vector_index.as_retriever(search_type="similarity", 
                                      search_kwargs={"k":10})
            
            # with open(f"{store_name}.pkl","wb") as f:
            #     pickle.dump(vectorstore,f)
        st.success("Embeddings Created")
        
        
        # if st.button("Generate Excel"):
        with st.spinner("Processing your queries"): 
            template2 = """You are an RFQ Analyst that helps Bid Team to find Relevant information in a RFQ. 
            You are given a RFQ document and a question.
            You need to find the answer to the question in the RFQ document.

            Give a single keyword answer but not sentences.
            If the answer is not in the document just say "I do not know".  
            {context}
            Question: {question}"""
            second_prompt = PromptTemplate.from_template(template2)
            qa_interface2 = RetrievalQA.from_chain_type(llm=llm,
                                                    retriever=retriever,
                                                    chain_type_kwargs={"prompt":second_prompt}, 
                                                    return_source_documents=True)
            response2=qa_interface2("What are the main systems given in RFQ")
            main_systems=response2['result']
            st.write(main_systems)
            delimiters=[',','.']
            for delimiter in delimiters:
                main_systems = " ".join(main_systems.split(delimiter))
            main_systems=main_systems.split()
            df=pd.DataFrame({"Systems":main_systems,"Architecture":"","Scope":"","BOM":"","Cabinet":""})
            df=df[df['Systems']!="system"]

            template3 = """You are an RFQ Analyst that helps Bid Team to find Relevant information in a RFQ. 
            You are given a RFQ document and a question.
            You need to find the answer to the question in the RFQ document.
            Give the answer in the below mentioned format
            For Example:
            Dimensions: 100mm (H) x 100mm (W) x 100mm (D)
            Material:Steel sheet
            Safety for indoor: IP 22  
            Safety for outdoor: IP 30 
            Thickness: 1.5 mm
            Color:RAL 7000
            Any other Specifications: bottom entry and Side mounting
            Don't Miss any Keywords in the given format. If you don't find answer for a particular keywords answer it as 'Not available'
            Also Give the answer under which heading does it belong to.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.  
            Always say "thanks for asking!" at the end of the answer.
            {context}
            Question: {question}"""
            second_prompt = PromptTemplate.from_template(template3)
            qa_interface3 = RetrievalQA.from_chain_type(llm=llm,
                                                    retriever=retriever,
                                                    chain_type_kwargs={"prompt":second_prompt}, 
                                                    return_source_documents=True)
            for system in df['Systems'].to_list():
                if system.lower()!='system':
                    response3=qa_interface3(f"What are the specifications for making a cabinet of {system}?")
                    # print(response3['result'])
                    df.loc[df['Systems']==system,['Cabinet']]=response3['result']

            template4 = """You are an RFQ Analyst that helps Bid Team to find Relevant information in a RFQ. 
            You are given a RFQ document and a question.
            You need to find the answer to the question in the RFQ document.
            Mention all the technical keywords in the architecture.
            For example:
            Q: Give me the architecture of the ESD system
            A: ESD system shall have ‘suitable modular redundant’ architecture (for example. Triple or Quadruple redundant) 
            utilising two-out-of-three or two-out-of-4D voting or any other equivalent redundant system architecture 
            with appropriate voting configuration to maintain SIL 3 integrity.

            In the above answer we have Technical wordings like Triple or Quadruple redundant,two-out-of-three or two-out-of-4D voting, redundant system architecture, SIL 3 integrity Menitoned in the architecture.
            Make sure in your answer also include these kind of Technical wordings

            Also Give the answer under which heading does it belong to.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.  
            Always say "thanks for asking!" at the end of the answer.
            {context}
            Question: {question}"""
            fourth_prompt = PromptTemplate.from_template(template4)
            qa_interface4 = RetrievalQA.from_chain_type(llm=llm,
                                                    retriever=retriever,
                                                    chain_type_kwargs={"prompt":fourth_prompt}, 
                                                    return_source_documents=True)
            for system in df['Systems'].to_list():
                if system.lower()!='system':
                    response4=qa_interface4(f"Give me the architecture of the {system}.")
                    print(response4['result'])
                    df.loc[df['Systems']==system,['Architecture']]=response4['result']

            template5 = """You are an RFQ Analyst that helps Bid Team to find Relevant information in a RFQ. 
            You are given a RFQ document and a question.
            You need to find the answer to the question in the RFQ document.
            Mention all the technical keywords in the Scope of Supply.
            Make sure in your answer also include Technical wordings.
            Summarize the lengthy sentences to short and crisp so the user can understand easily.

            Also Give the answer under which heading does it belong to.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.  
            Always say "thanks for asking!" at the end of the answer.
            {context}
            Question: {question}"""
            fifth_prompt = PromptTemplate.from_template(template5)
            qa_interface5 = RetrievalQA.from_chain_type(llm=llm,
                                                    retriever=retriever,
                                                    chain_type_kwargs={"prompt":fifth_prompt}, 
                                                    return_source_documents=True)
            for system in main_systems:
                response5=qa_interface5(f"Give me the scope of supply for {system}.")
                print(response5['result'])
                df.loc[df['Systems']==system,['Scope']]=response5['result']

            template6 = """You are an RFQ Analyst that helps Bid Team to find Relevant information in a RFQ. 
            You are given a RFQ document and a question.
            You need to find the answer to the question in the RFQ document.
            Extract the Bill of Materials from the give document
            For example:
            Redundancy controller
            Redunduncay IO Cards
            Galvanic Isolation
            FTA Board
            Modbus
            Profibus

            Mostly the Bill of Materials will be of hardware components. Extract those in a list.
            Also Give the answer under which heading does it belong to.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.  
            Always say "thanks for asking!" at the end of the answer.
            {context}
            Question: {question}"""
            sixth_prompt = PromptTemplate.from_template(template6)
            qa_interface6 = RetrievalQA.from_chain_type(llm=llm,
                                                    retriever=retriever,
                                                    chain_type_kwargs={"prompt":sixth_prompt}, 
                                                    return_source_documents=True)

            for system in main_systems:
                response6=qa_interface6(f"What are the Bill of Materials for {system}.")
                print(response6['result'])
                df.loc[df['Systems']==system,['BOM']]=response6['result']


            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')


            csv = convert_df(df)

            if st.download_button(
            "Press to Download",
            csv,
            file_name="RFQ-summary.csv",
            mime="text/csv",
            key='download-csv'
            ):
                st.success("File is downloaded successfully")
            end_time=time.time()
            st.write(f"Time Taken: {end_time-start_time} seconds")
                


if __name__=="__main__":
    main()
