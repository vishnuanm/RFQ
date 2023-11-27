# Importing the required libraries
import openai
import os
import uuid
import time
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Importing the Azure libraries
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient


class PDFSearchEngine:
    def __init__(self):
        """
        The function initializes the necessary variables and sets up the Azure OpenAI, Azure Search
        keys and Azure OpenAI Embeddings model.
        """
        load_dotenv()

        # Setting up the Azure OpenAI and Azure Search Keys
        self.deployment_name = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
        openai.api_type = "azure"
        openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        self.AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")


    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def generate_embeddings(self, text):
        """
        The generate_embeddings function takes in a string of text and returns an embedding vector.
        The function uses the OpenAI API to generate the embeddings.
        
        :param self: Represent the instance of the class
        :param text: Pass the text to be embedded
        :return: Embedding vector
        """
        response = openai.Embedding.create(input=text, engine=self.deployment_name)
        embeddings = response['data'][0]['embedding']
        return embeddings


    def upload_embeddings(self, doc_chunks, index_name):
        """
        The upload_embeddings function takes in a list of document chunks and an index name.
        It then generates embeddings for each chunk, and uploads the documents along with their embeddings and metadata into the vector store.
        
        :param self: Bind the method to an object
        :param doc_chunks: Pass in the documents that are to be uploaded
        :param index_name: Specify the name of the index to be created
        :return: search_client object
        """
        doc_chunks_embedding = []
        for doc in doc_chunks:
            doc_chunks_embedding.append({'documentID':str(uuid.uuid4()), "content":doc.page_content, \
                        "embedding":self.generate_embeddings(doc.page_content),\
                        "source_document_name":doc.metadata['source'], "page_number": int(doc.metadata['page'])
                                    })

        # Uploading the documents along with the embeddings into the vector store
        search_client = SearchClient(endpoint=self.AZURE_SEARCH_ENDPOINT, index_name=index_name, credential=AzureKeyCredential(self.AZURE_SEARCH_KEY))
        result = search_client.upload_documents(doc_chunks_embedding)
        return search_client
