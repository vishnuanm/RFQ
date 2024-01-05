import os
import openai
import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from streamlit_extras.add_vertical_space import add_vertical_space
# from PIL import Image
import PIL.Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
import fitz
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd
from pandas.io.excel._xlsxwriter import XlsxWriter
import re 
import json
import AzureAPP
import docx2txt
import streamlit_ext as ste
import time
from streamlit_chat import message
from tenacity import retry, stop_after_attempt, wait_random_exponential
from spire.doc import *
from spire.doc.common import *



openai.api_type = "azure"
openai.api_base = "https://rfq-abb.openai.azure.com/"
openai.api_key = "192d27f4fc584d76abd8a5eb978dcedf"
openai.api_version = "2023-03-15-preview"


with st.sidebar:
    st.title("Ask the Doc")
    st.markdown("This is a LLM powered chatbot that can answer your questions about RFQs")
    add_vertical_space(5)
    st.write("Made by")
    image = PIL.Image.open('./Images/logo.png')
    st.sidebar.image(image, width=200)

class RichExcelWriter(XlsxWriter):
    def __init__(self, *args, **kwargs):
        super(RichExcelWriter, self).__init__(*args, **kwargs)

    def _value_with_fmt(self, val):
        if type(val) == list:
            return val, None
        return super(RichExcelWriter, self)._value_with_fmt(val)

    def _write_cells(self, cells, sheet_name=None, startrow=0, startcol=0, freeze_panes=None):
        sheet_name = self._get_sheet_name(sheet_name)
        if sheet_name in self.sheets:
            wks = self.sheets[sheet_name]
        else:
            wks = self.book.add_worksheet(sheet_name)
            wks.set_column(0, 0, 40)
            wks.set_column(1, 5, 50)
            #add handler to the worksheet when it's created
            wks.add_write_handler(list, lambda worksheet, row, col, list, style: worksheet._write_rich_string(row, col, *list))
            self.sheets[sheet_name] = wks
        super(RichExcelWriter, self)._write_cells(cells, sheet_name, startrow, startcol, freeze_panes)

def create_excel_with_formatting_local(df, filename, sheet_name):
    """
    The create_excel_with_formatting function takes a DataFrame, filename, and sheet_name as input.
    It then creates an Excel file with the specified name and adds a worksheet to it with the specified name.
    The function then applies bold formatting to any text in the DataFrame that is surrounded by HTML <b></b> tags.
    
    :param df: Pass in the dataframe that will be converted to excel
    :param filename: Name the excel file that will be created
    :param sheet_name: Name the sheet in the excel file
    :return: A pandas excelwriter object
    """
    writer = RichExcelWriter(filename)
    workbook = writer.book
    bold = workbook.add_format({'bold': True})


    # Function to convert HTML bold tags to Excel bold formatting
    def convert_html_tags(text):
        """
        The convert_html_tags function takes a string as input and returns the same string with HTML tags converted to Excel formatting.
        
        :param text: Pass in the text that will be formatted
        :return: A list of formatted strings
        """
        if isinstance(text, float):
            return ' '
        if '<b>' not in text:
            return text
        parts = re.split(r'(<b>|</b>)', text)
        formatted_parts = [bold if part == '<b>' else part for part in parts if part != '</b>']
        return formatted_parts


    # Apply the function to each cell in the DataFrame
    for col in df.columns:
        df[col] = df[col].apply(convert_html_tags)

    output = df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.close()
    return output

def generate_response(llm, retriever_data, prompt_template, query_text):    
    
    qa_interface2 = RetrievalQA.from_chain_type(llm=llm,
                                                retriever=retriever_data,
                                                chain_type_kwargs={"prompt":prompt_template},  
                                                return_source_documents=True)
    return qa_interface2(query_text)['result']


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def generate_embeddings(uploaded_files,openai_api_key, openai_api_base, openai_api_version):
    embeddings = OpenAIEmbeddings(deployment_id="rfq-embeddings", chunk_size=1,openai_api_key=openai_api_key, openai_api_base=openai_api_base, openai_api_version=openai_api_version)
    text = ""
    for file in uploaded_files:
        file_type = os.path.splitext(file.name)[1]
        if file_type == ".pdf" or file_type == ".PDF":
            pdf_file = fitz.open(stream=file.getvalue(), filetype="pdf")
            for page in pdf_file:
                text += page.get_text()
        elif file_type ==".docx":
            text += docx2txt.process(file)
        elif file_type==".doc":
            with open(file.name,"wb") as f:
                f.write(file.getbuffer())
            document = Document()
            # Load a Word DOC file
            document.LoadFromFile(file.name)
            # Save as .docx file
            document.SaveToFile("ToDocx.docx", FileFormat.Docx2016)
            text += docx2txt.process("ToDocx.docx")
            os.remove("ToDocx.docx")

            

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    splits = text_splitter.split_text(text)

    directory="vector_store"
    vectorstore = FAISS.from_texts(splits, embedding=embeddings)
    vectorstore.save_local(directory)
    vector_index = FAISS.load_local('vector_store', 
                                    OpenAIEmbeddings(openai_api_key = openai.api_key,
                                                    deployment_id = "rfq-embeddings"))
    retriever = vector_index.as_retriever(search_type="similarity", 
                                            search_kwargs={"k":3})
    return vectorstore, retriever

def clear_chat():
    st.session_state['generated'] = []
    st.session_state['past'] = []
    del st.session_state.generated
    del st.session_state.past

def main():
    st.header("RFQ Extraction")
    azure = st.checkbox('Azure')

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if azure:
        st.write("Using Azure")
        AzureAPP.main()
    else:
        
        start_time = time.time()
        embeddings = OpenAIEmbeddings(deployment_id="rfq-embeddings", chunk_size=1,openai_api_key=openai.api_key, openai_api_base=openai.api_base, openai_api_version=openai.api_version)
        uploaded_files = []
    
        # Loop through the uploaded files and append them to the list
        for file in st.file_uploader("Upload your files here", accept_multiple_files=True,type=["pdf", "docx"]):
            uploaded_files.append(file)

        f=open("keywords.json")
        data = json.load(f)
        arch_keywords=data['Keywords']['Architecture']
        # Scope_keywords=data['Keywords']['Scope']
        system_keywords=data['Keywords']['System']
        bom=data['Keywords']['BOM']
        not_bom=data['Keywords']['Not_BOM']
        main_sys=data['Keywords']['Main_System']

        if 'epoch' not in st.session_state:
            st.session_state.epoch = 1
            st.session_state.vectorstore = None
            st.session_state.retriever = None
            llm = AzureChatOpenAI(temperature=0.3,deployment_name="rfq8k", openai_api_key=openai.api_key, openai_api_base=openai.api_base, openai_api_version=openai.api_version)
            st.session_state.llm = llm

        if uploaded_files:
            if st.session_state.epoch == 1:
                st.session_state.vectorstore, st.session_state.retriever = generate_embeddings(uploaded_files,openai.api_key, openai.api_base, openai.api_version)
                st.success("Embeddings Created")
                st.session_state.epoch += 1
            
            # search_container = st.container()
            # Input For Any Further Query
            styl = f"""
            <style>
                .stTextInput {{
                position: fixed;
                bottom: 3rem;
                }}
            </style>
            """
            st.markdown(styl, unsafe_allow_html=True)

            chat_template = """Act as a Request for Quote (RFQ) Analyst that helps Bid Team to find Relevant information in a RFQ. 
            If the answer is not in the document just say "I do not know".  
            {context}
            Question: {question}"""
            chat_prompt = PromptTemplate.from_template(chat_template)
            
            if "my_text" not in st.session_state:
                st.session_state.my_text = ""

            def submit():
                st.session_state.my_text = st.session_state.widget
                st.session_state.widget = ""
            st.text_input("Any further query?",key="widget", on_change=submit, placeholder ="Search here...")
            my_text = st.session_state.my_text
            if my_text:
                with st.spinner("Processing your request..."):
                    st.session_state.past.append(my_text)
                    generated_response = generate_response(st.session_state.llm, st.session_state.retriever, chat_prompt,my_text )
                    st.session_state.generated.append(generated_response)
                    # st.markdown(generated_response)

            if st.session_state['generated']:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user',avatar_style="initials",seed="U")
                    message(st.session_state["generated"][i], key=str(i),avatar_style="initials",seed="AI")

            st.sidebar.button("Clear Chat History", on_click=clear_chat,key="clear_chat")
                
            styl = f"""
            <style>
                .button {{
                position: fixed;
                bottom: 1rem;
                }}
            </style>
            """
            st.markdown(styl, unsafe_allow_html=True)
            if st.button("Generate Excel"):
                with st.spinner("Processing your queries"): 
                    main_system_template = """Act as a Request for Quote (RFQ) Analyst that helps Bid Team to find Relevant information in a RFQ. 
                    You are given a RFQ document and a question.
                    You need to find the answer to the question in the RFQ document.
                    Give the answer in a single Technical Keywords instead of sentences. 
                    Examples of systems you find the document are mentioned in triple backticks ```{keywords}```
                    If the answer is not in the document just say "I do not know".  
                    {context}
                    Question: {question}"""
                    main_system_prompt = PromptTemplate.from_template(main_system_template,partial_variables={"keywords":main_sys})
                    input_query="What are the main systems in the RFQ?"
                    main_systems=generate_response(st.session_state.llm, st.session_state.retriever, main_system_prompt,input_query )
                    st.write(main_systems)
                    main_systems=main_systems.split(',')
                    df=pd.DataFrame({"Systems":main_systems,"Scope":"","Architecture":"","BOM":"","Cabinet":"","IO":""})

                    cabinet_template = """Act as a Request for Quote (RFQ) Analystt that helps Bid Team to find and summarize the Relevant information in the given document.
                    Extract the answer in the below mentioned format
                    For Example:
                    Dimensions: 100mm (H) x 100mm (W) x 100mm (D)
                    Material:Steel sheet
                    Safety for indoor: IP 22  
                    Safety for outdoor: IP 30 
                    Thickness: 1.5 mm
                    Color:RAL 7000
                    Hazardous Area Classification: (Can be any of "Zone0", "Zone1", "Zone2", "Zone20", "Zone21", "Zone22", "Safe Area",if not mentioned, then "N/A")
                    Certification: (Like "UL","IECEx","ATEX")
                    Panel Mounted HMI along with inches: (Can be either "Yes 12 inches","No" )
                    Matrix/Mimc Console: (Can be either "Yes","No" )

                    Don't Miss any Keywords in the given format. If you don't find answer for a particular keywords answer it as 'Not available'
                    Highlight the keywords present in the output. Surround the keywords with <b> tags. Example: Like this, The quick <b>keyword1</b> jumps over the lazy <b>keyword2</b>."
                    Also Give the answer under which heading does it belong to.
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.  
                    {context}
                    Question: {question}"""
                    cabinet_prompt = PromptTemplate.from_template(cabinet_template)
                    for system in df['Systems'].to_list():
                        input_query=f"What are the specifications for making a cabinet of {system.strip()}?"
                        cabinet_response=generate_response(st.session_state.llm, st.session_state.retriever, cabinet_prompt,input_query )
                        df.loc[df['Systems']==system,['Cabinet']]=cabinet_response

                    arch_template = """Act as a Request for Quote (RFQ) Analyst that helps Bid Team to find and summarize the Relevant information in the given document. 
                    Give me the output in bullet points.
                    Cover the technical keywords given in triple backticks while giving the answer.
                    ```{keywords}```
                    Surround all the Technical keywords with <b> tags. Example: Like this, The architecture will support <b>hot mode (online)</b>  replacement of faulty modules without degradation of system functionality, <b>SIL 3 integrity</b> , and high availability."
                    Give me the page number also from where you have fetched the answer from, if there are multiple pages give all of them.
                    Also Give the answer under which heading does it belong to.
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.  
                    {context}
                    Question: {question}"""
                    arch_prompt = PromptTemplate.from_template(arch_template,partial_variables={"keywords":arch_keywords})
                    input_query=f"Give the system architecture which includes some of the following elements like {system_keywords} "
                    arch_reponse=generate_response(st.session_state.llm, st.session_state.retriever, arch_prompt,input_query )
                    df.loc[df['Systems']==main_systems[0],['Architecture']]=arch_reponse

                    scope_template = """You are an Analyst and information extractor for the provided 'Request for Quote (RFQ)' document and your task is to identify and provide relevant summarized answer in just one or two lines from the provided RFQ document for the asked question instead of giving a lengthy or detailed response. 
                    Surround all the technical keywords with <b> tags, for Example: "Design and supply of the <b>PCS Auxiliary Cabinets</b>"

                    Give me the page number also from where you have fetched the answer from, if there are multiple pages give all of them.
                    Also if the answer is fetched from a paragraph/section with heading give the heading/section name.
                    If the complete answer is not in the document just give reply as "Couldn't find the information in the RFQ", don't try to make up an answer.
                    {context}
                    Question: {question}"""
                    scope_prompt = PromptTemplate.from_template(scope_template)
                    for system in main_systems:
                        input_query=f"Summarize the scope for {system.strip()} in just one or two lines."
                        scope_response=generate_response(st.session_state.llm, st.session_state.retriever, scope_prompt,input_query )
                        df.loc[df['Systems']==system,['Scope']]=scope_response

                    bom_template = """Act as a Request for Quote (RFQ) Analyst that helps Bid Team to find and summarize the Relevant information in the given document.
                    Extract the Comprehensive list of  components, devices,Instruments and software that constitute a control system by using keywords listing triple backticks ```{bom}```. Also include the quantities if mentioned.
                    Extract those in a list.
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.  
                    {context}
                    Question: {question}"""
                    bom_prompt = PromptTemplate.from_template(bom_template,partial_variables={"bom":bom})

                    for system in main_systems:
                        input_query=f"Give the Bill of Materials (BOM) in {system.strip()}."
                        bom_response=generate_response(st.session_state.llm, st.session_state.retriever, bom_prompt,input_query )
                        df.loc[df['Systems']==system,['BOM']]=bom_response

                    io_template = """Act as a Request for Quote (RFQ) Analyst that helps Bid Team to find and summarize the Relevant information in the given document.Extract the following
                    What is the Minimum Channel requirement for AI,AO,DI,DO Module? (Can be one of "8","16","32")
                    Which system have Redundant IO? (Can be one of  "ESD","FGS")
                    What are the Type of IO Modules? (Can be one of  "IS(Intrinsically Safe)", "Non-IS", "SIL Non-SIL" (High Integrity))
                    Is HART Required for AI/AO? (Can be one of  "Yes","No")
                    What is the Resolution for Each Module? (Can be one of "16bits","12 bits") 
                    Line Monitoring (Line Fault Detection) (Can be one of AI- "Yes","No", DI- "Yes","No", DO- "Yes","No", AO- "Yes","No")
                    Is Galvanic Isolation present ?(Can be one of  "Yes","No")
                    Does SOE (Sequence of Event) Required (Can be one of  "Yes","No")
                    Is it Universal/Configurable/Smart IO Module (Can be one of "Yes","No")
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.  
                    {context}
                    Question: {question}"""
                    io_prompt = PromptTemplate.from_template(io_template)

                    for system in main_systems:
                        input_query=f"I/O Specification for {system.strip()}."
                        io_response=generate_response(st.session_state.llm, st.session_state.retriever, io_prompt,input_query )
                        df.loc[df['Systems']==system,['IO']]=io_response
                    
                    # template7 = """You are an RFQ Analyst that helps Bid Team to find Relevant information in a RFQ. 
                    # You are given a RFQ document and a question.
                    # You need to find the answer to the question in the RFQ document.
                    # Give the answer in a single word.
                    # For Example:
                    # SIL 3 Certificate
                    # IEC 61508
                    # Exida
                    # TUV
                    # Training Certificate


                    # Highlight the Hardware components present in the output. Surround the Hardware components with <b> tags. Example: Like this, The quick <b>keyword1</b> jumps over the lazy <b>keyword2</b>."
                    # Also Give the answer under which heading does it belong to.
                    # If you don't know the answer, just say that you don't know, don't try to make up an answer.  
                    # Always say "thanks for asking!" at the end of the answer.
                    # {context}
                    # Question: {question}"""
                    # seventh_prompt = PromptTemplate.from_template(template7)
                    # qa_interface7 = RetrievalQA.from_chain_type(llm=llm,
                    #                                         retriever=retriever,
                    #                                         chain_type_kwargs={"prompt":seventh_prompt}, 
                    #                                         return_source_documents=True)

                    # for system in main_systems:
                    #     response7=qa_interface7(f"What are the Documents or Certificates in the RFQ.")
                    #     print(response7['result'])
                    #     df.loc[df['Systems']==system,['Documentation']]=response7['result']

                    


                    # @st.cache_data
                    # def convert_df(df):
                    #     return df.to_csv(index=False).encode('utf-8')


                    # csv = convert_df(df)
                output = create_excel_with_formatting_local(df, 'rfq_output.xlsx', 'output')
                with open('rfq_output.xlsx', 'rb') as f:
                        file_data = f.read()
                st.write("Done! Click below to download the Excel output.")
                ste.download_button(
                label="Download Excel File",
                data=file_data,
                file_name="rfq_output.xlsx")
                end_time=time.time()
                st.write(f"Time Taken: {end_time-start_time} seconds")

            

            
                        


if __name__=="__main__":
    main()
