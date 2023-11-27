# Importing the required libraries
import io
import os
import re
import sys
import json
import pickle
import pandas as pd
sys.path.append(".")
import openai
import streamlit as st

# Importing the Azure libraries
from azure.search.documents.models import (QueryAnswerType, QueryCaptionType, QueryLanguage, QueryType, RawVectorQuery)

# Importing the custom packages and modules
from .search.embedding import PDFSearchEngine
from .search.search_index import PDFSemanticSearch
from .parse.document_ai_pdf_loader import BlobStorageProcessor, chunk_documents
from .parse.excel_formatter import RichExcelWriter, create_excel_with_formatting

# Impoting libraries for Prompt
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser, PydanticOutputParser
from langchain.chat_models import AzureChatOpenAI
from pydantic import BaseModel, Field 
from typing import List

import functools
import time

import asyncio
import aiohttp
import concurrent.futures


# def timer(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         start_time = time.perf_counter()
#         value = func(*args, **kwargs)
#         end_time = time.perf_counter()
#         run_time = end_time - start_time
#         st.write("Finished {} in {} secs".format(repr(func.__name__), round(run_time, 3)))
#         return value

#     return wrapper


class RFQExtractionAzure:
    def __init__(self, folder_name, search_index_name, output_file_path, use_semantic_configuration=False):
        """
        The function initializes an object with directory path, search index name, and a flag for using
        semantic configuration.
        
        :param directory_path: The directory path is the location of the directory where the files to be
        searched are stored. It is a string that specifies the path to the directory on the file system
        :param search_index_name: The `search_index_name` parameter is the name of the search index that
        will be used for searching and retrieving documents. It is typically a string that identifies a
        specific index within a search service or engine
        :param use_semantic_configuration: A boolean flag indicating whether to use semantic
        configuration for the search index. If set to True, the search index will be configured to
        understand the meaning of the documents and provide more accurate search results. If set to
        False, a basic keyword-based search will be used, defaults to False (optional)
        """
        self.folder_name = folder_name
        self.search_index_name = search_index_name
        self.use_semantic_configuration = use_semantic_configuration
        self.openai_engine = "rfq"
        self.pdf_search = None
        self.search_client = None
        self.output_file_path = output_file_path

        # Initiliasing the AzureChatOpenAI and required parameters
        self.model = AzureChatOpenAI(   
                                    openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                                    deployment_name=os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME"),
                                    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                                    openai_api_type="azure",
                                    temperature = 0.3
                                    )

    def parse_documents_and_create_chunks(self, chunk_size, chunk_overlap):
        """
        The function loads PDF documents from a folder present in given container, 
        processes them using a document processor - split them into pages,
        and then return the chunks based on the specified size.

        :return: chunks of all the PDF documents present in the given folder
        """
        
        pdf_processor = BlobStorageProcessor()
        all_document_pages = pdf_processor.process_pdfs_from_folder(self.folder_name)
        all_document_chunks = chunk_documents(all_document_pages, chunk_size, chunk_overlap)
        return all_document_chunks

    def create_search_index(self):
        """
        The function creates a vector search index for PDF documents using vector and semantic search configuration

        :return: None object
        """

        self.pdf_search = PDFSemanticSearch(self.search_index_name, self.use_semantic_configuration)
        self.pdf_search.create_search_index()


    def upload_embeddings(self, chunks):
        """
        The function `upload_embeddings` uploads chunks of data to a search engine index using a PDF
        search engine.
        
        :param chunks: The "chunks" parameter is a dictionary consist of chunks of text and their corresponding embeddings and the metadata. 
        """
        
        self.pdf_search_engine = PDFSearchEngine()
        self.search_client = self.pdf_search_engine.upload_embeddings(chunks, self.search_index_name)


    def perform_vector_semantic_search(self, context_query, num_doc_chunks):
        """
        The function performs a vector search using a given query and returns the search results.
        
        :param vquery: The `vquery` parameter is the query vector that you want to use for vector
        search. It represents the vectorized representation of the query text or document that you want
        to search for
        :return: the search results based on a vector query. The results include the content and the metadata of the
        document chunks that match the query.
        """
       
        vector_query = RawVectorQuery(vector=self.pdf_search_engine.generate_embeddings(context_query), k=num_doc_chunks, fields="embedding")
        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["content"],
            query_type=QueryType.SEMANTIC,
            query_language=QueryLanguage.EN_US,
            semantic_configuration_name='my-semantic-config',
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE
        )
        return results


    def prompt_template_creation(self, query, input_text, system, keywords, format_instructions=None, output_parser=None):
        """
        The prompt_template_creation function is used to create a prompt template for the user.
            The function takes in the following parameters:
                query (str): The question that will be asked of the user. 
                input_text (str): A string containing text that will be used as an input variable for the prompt. 
                system (str): A string containing text that will be used as an input variable for the prompt. 
                keywords (list[str]): A list of strings containing keywords or phrases, which are passed to the prompt.

        :param query: Create the prompt template
        :param input_text: Replace the placeholder variable in the query
        :param system: Determine which system the user wants to search
        :param keywords: Replace the placeholder variable in the query
        :param format_instructions: Specify the format of the input text
        :param output_parser: Parse the output of the user's response
        :return: A prompt object with the query and input variables
        """
        # Creating the prompt template
        if system == '':
            prompt_template = ChatPromptTemplate(
            messages=[HumanMessagePromptTemplate.from_template(query)],\
                        input_variables=["input_text", "system", "keywords"],
                        partial_variables={"format_instructions": format_instructions},
                        output_parser=output_parser)
        else:
            prompt_template = ChatPromptTemplate(
            messages=[HumanMessagePromptTemplate.from_template(query)],\
                        input_variables=["input_text", "system", "keywords"], 
                        partial_variables={"format_instructions": format_instructions})
        
        # Replacing the placeholder variables
        prompt = prompt_template.format_messages(input_text=input_text, system=system, keywords=keywords, format_instructions=format_instructions)
        
        return prompt


    def highlight_keywords(self, text, keywords):
        """
        The highlight_keywords function takes a string of text and a comma-separated list of keywords.
        It returns the original text with all instances of any keyword wrapped in <b></b> tags.
        
        :param self: Access the instance of the class
        :param text: Pass the text that needs to be highlighted
        :param keywords: Specify the list of keywords that should be highlighted
        :return: The text with the keywords highlighted
        """
        # Split the keywords string into a list of keywords
        keywords = [keyword.strip() for keyword in keywords.split(',')]

        # Create a regular expression pattern that matches any keyword that is not already highlighted
        pattern = r"(?<!<b>)\b(" + "|".join(re.escape(keyword) for keyword in keywords) + r")\b(?!<\/b>)"

        # Define a replacement function that adds <b> tags around the matched keyword
        def replacement(match):
            return '<b>' + match.group() + '</b>'

        # Replace all non-highlighted instances of any keyword
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    # def multi_query_extract_func(self, systems_lst, query, input_text, format_instructions, keywords):
    #     """
    #     The multi_query_extract_func function is used to query the OpenAI model for multiple systems.
    #         It takes in a list of systems, a query, input text and keywords as parameters.
    #         The function returns a dictionary with system names as keys and their corresponding responses from the OpenAI model.
        
    #     :param systems_lst: Specify the subsystems for which we want to extract information
    #     :param query: Create the prompt for openai model
    #     :param input_text: Pass the input text to the function
    #     :param format_instructions: Specify the format of the output text
    #     :param keywords: Highlight the keywords in the response
    #     :return: A dictionary with the system name as key and the response from openai model as value
    #     """
    #     # Dictionary to store the information corresponding to each system
    #     result_dict = {}
        
    #     # Looping through each subsystem in the list and querying openai model
    #     for system in systems_lst:
    #         prompt = self.prompt_template_creation(query, input_text, system, keywords, format_instructions)
    #         response = self.model(prompt)
    #         result_dict[system] = self.highlight_keywords(response.content, keywords)
            
    #     return result_dict

    # Function to be executed in parallel
    def query_model(self, system, prompt_template_creation, model, highlight_keywords, query, input_text, keywords, format_instructions):
        prompt = prompt_template_creation(query, input_text, system, keywords, format_instructions)
        response = model(prompt)
        return system, highlight_keywords(response.content, keywords)

    def multi_query_extract_func(self, systems_lst, query, input_text, format_instructions, keywords):
        """
        The multi_query_extract_func function is used to query the OpenAI model for multiple systems.
        It takes in a list of systems, a query, input text and keywords as parameters.
        The function returns a dictionary with system names as keys and their corresponding responses from the OpenAI model.
        """
        # Dictionary to store the information corresponding to each system
        result_dict = {}

        # Use ProcessPoolExecutor to execute each system query in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.query_model, system, self.prompt_template_creation, self.model, self.highlight_keywords, query, input_text, keywords, format_instructions) for system in systems_lst}
            for future in concurrent.futures.as_completed(futures):
                system, response = future.result()
                result_dict[system] = response

        return result_dict


    def run_query(self, query_data, num_doc_chunks, systems_lst=None, output_parser=None, format_instructions=None, query_name=None):
        """
        The run_query function is used to query the model.
            Args:
                query_data (dict): A dictionary containing the keywords and input queries. The keys are 'keywords' and 'query'. 
                systems_lst (list): A list of systems that you want to extract data from. If None, then it will use all available systems in your model's config file. 
        
        :param query_data: Pass the query and keywords to the function
        :param systems_lst: Pass a list of systems to the function
        :param output_parser: Parse the output of the model
        :param format_instructions: Pass the instructions to the model
        :return: A dictionary of results
        """
        # Context and Input queries
        #context_query = query_data["context_query"]
        query = query_data['query']
        keywords = query_data['keywords']

        # Perform vector search and get document chunks using context_query
        results = self.perform_vector_semantic_search(keywords, num_doc_chunks)
        input_text = " ".join(result['content'] for result in results)

        if systems_lst is None:
            # Creating the prompts and querying the model - passing null to system and keywords
            prompt = self.prompt_template_creation(query, input_text, '', keywords, format_instructions, output_parser)
            response = self.model(prompt)
            return output_parser.parse(response.content)

        # Format instructions
        format_instructions = f"Highlight the following list of keywords present in the output. Surround the keywords with <b> tags - [{keywords}]. Example: Like this, The quick <b>keyword1</b> jumps over the lazy <b>keyword2</b>.".format(keywords)

        # Architecture - single prompt
        if query_name == "Architecture":
            result_dict = {system: '' for system in systems_lst}
            prompt = self.prompt_template_creation(query, input_text, str(systems_lst), keywords, format_instructions)
            response = self.model(prompt)
            result_dict[query_name] = self.highlight_keywords(response.content, keywords)
            return result_dict
        
        # Querying the model for each system
        result_dict = self.multi_query_extract_func(systems_lst, query, input_text, format_instructions, keywords)
        return result_dict

    def process_queries(self, queries, num_doc_chunks):
        """
        The process_queries function is used to extract the information from the PDFs.
            It takes in a dictionary of queries and returns a tuple of dictionaries containing 
            all the extracted information.
        
        :param queries: Pass the queries to be run in the function
        :return: A tuple of dictionaries
        """
        # Extracting the subsytems
        systems_lst = self.run_query(queries["systems_query"], num_doc_chunks, None, CommaSeparatedListOutputParser(), \
                                     CommaSeparatedListOutputParser().get_format_instructions())


        # Extracting the Scope, Architecture and Bill of Materials Requirements
        scope_output_dict = self.run_query(queries["scope_query"], num_doc_chunks, systems_lst)
        arc_output_dict = self.run_query(queries["architecture_query"],num_doc_chunks, systems_lst, query_name="Architecture")
        bom_output_dict = self.run_query(queries["bom_query"], num_doc_chunks, systems_lst)
        #cabinet_output_dict = self.run_query(queries["cabinet_query"], num_doc_chunks, systems_lst)
        # doc_output_dict = self.run_query(queries["documentation_query"], num_doc_chunks, systems_lst)
        # eng_output_dict = self.run_query(queries["engineering_query"], num_doc_chunks, systems_lst)
        # testing_output_dict = self.run_query(queries["testing_query"], num_doc_chunks, systems_lst)
        # siteservice_output_dict = self.run_query(queries["site_services_query"], num_doc_chunks, systems_lst)

        #arc_output_dict = None
        #bom_output_dict = None
        cabinet_output_dict = None
        doc_output_dict = None
        eng_output_dict = None
        testing_output_dict = None
        siteservice_output_dict = None

        
        return scope_output_dict, arc_output_dict, bom_output_dict, cabinet_output_dict, doc_output_dict, eng_output_dict, testing_output_dict, siteservice_output_dict


    def excel_output_creation(self, output_path, scope_output, arc_output, bom_output, cabinet_output, doc_output, eng_output, testing_output, siteservice_output):
        """
        The excel_output_creation function is used to create the excel output file.
            It takes in the following parameters:
                - self (object) : The object of class PDFSemanticSearch. 
                - output_path (string) : The path where the excel file needs to be created. 
                - scope_output (dictionary): A dictionary containing all systems and their corresponding scopes as key-value pairs. 
        
        :param self: Represent the instance of the class
        :param output_path: Define the path where the output excel file will be saved
        :param scope_output: Populate the first column of the dataframe
        :param arc_output: Map the architecture dataframe to the result_df dataframe
        :param bom_output: Populate the bill of materials column in the excel output file
        :param cabinet_output: Populate the cabinet column in the excel output
        :param doc_output: Map the documentation output to the dataframe
        :param eng_output: Populate the engineering column in the excel output
        :param testing_output: Map the testing output to the dataframe
        :param siteservice_output: Map the site services column in the excel output
        :return: A dataframe
        """
        # Defining the dataframe - in required format
        result_df = pd.DataFrame(scope_output.items(), columns =['System', 'Scope'])

        # Populating the dataframe
        result_df.loc[len(result_df),'System'] = "Architecture"
        result_df['Architecture'] = result_df['System'].map(arc_output)
        result_df['Bill Of Materials'] = result_df['System'].map(bom_output)
        #result_df['Cabinet'] = result_df['System'].map(cabinet_output)
        # result_df['Documentation'] = result_df['System'].map(doc_output)
        # result_df['Engineering'] = result_df['System'].map(eng_output)
        # result_df['Testing'] = result_df['System'].map(testing_output)
        # result_df['Site Services'] = result_df['System'].map(siteservice_output)

        return result_df


    def rfq_extraction(self):
        """
        The rfq_extraction function is the main function that performs all the steps required to extract RFQ information from a set of documents.
        
        :param self: Bind the method to an object
        :return: A dataframe with the results
        """
        # Load the Queries - from JSON file
        st.write("\n Reading Queries from JSON file \n")
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        with open(os.path.join(__location__, 'queries.json'), 'r') as file:
            queries = json.load(file)

        # Load documents
        st.write("Parsing RFQ PDF's and Creating Document Chunks \n")
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        with open(os.path.join(__location__, 'params.json'), 'r') as file:
            params = json.load(file)
        chunks = self.parse_documents_and_create_chunks(params["chunk_size"], params["chunk_overlap"])

        # Create and initialize the search index
        st.write("Creating Vector Store in Azure \n")
        self.create_search_index()

        # Upload embeddings to the Azure Search index
        st.write("Uploading Embeddings with metadata to the Vector Store \n")
        self.upload_embeddings(chunks)

        # Processing the queries
        st.write("Processing the Pre-defined Queries \n")
        scope_output, arc_output, bom_output, cabinet_output, doc_output, eng_output, testing_output, siteservice_output = self.process_queries(queries, params["num_doc_chunks"])

        # Storing the result in Excel file
        st.write("Exporting the Responses in an Excel file \n")
        result_df = self.excel_output_creation(self.output_file_path, scope_output, arc_output, bom_output, cabinet_output, doc_output, eng_output, testing_output, siteservice_output)

        # Cleaning up the index
        self.pdf_search.delete_search_index()
        st.write("Index Cleaned Up \n")

        # Cleaining up the folder in Azure Blob Container
        BlobStorageProcessor().delete_folder(self.folder_name)
        st.write("Storage Cleaned Up \n")
        
        return result_df

if __name__ == "__main__":
    folder_name = "ages_1103"
    search_index_name = "vector-search-index1152"
    use_semantic_configuration = False
    #use_buffer_memory = False
    output_file_path = "rfq_extraction/output.xlsx"

    rfq_extraction_obj = RFQExtractionAzure(folder_name, search_index_name, output_file_path, use_semantic_configuration)
    response = rfq_extraction_obj.rfq_extraction()