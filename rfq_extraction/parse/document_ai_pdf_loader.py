import os
from datetime import datetime, timedelta
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from langchain.document_loaders.pdf import DocumentIntelligenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from azure.storage.blob import BlobServiceClient, ContainerClient
from dotenv import load_dotenv
from azure.storage.blob import BlobSasPermissions, generate_blob_sas

# Load environment variables
load_dotenv()

class DocumentProcessor:
    """ This Class is used to parse PDF documents present in the local storage 
    """
    def __init__(self):
        """
        The function initializes the document analysis client.
        """
        self.document_analysis_client = self.initialize_document_analysis_client()


    def initialize_document_analysis_client(self):
        """
        The function initializes a DocumentAnalysisClient object using the Azure Document Intelligence
        endpoint and key from environment variables.

        :return: a DocumentAnalysisClient object.
        """
        # Get the Azure Document Intelligence endpoint and key from environment variables
        document_ai_endpoint = os.getenv("AZURE_DOCUMENTAI_ENDPOINT")
        document_ai_key = os.getenv("AZURE_DOCUMENTAI_KEY")
        
        # Create and return a DocumentAnalysisClient object
        return DocumentAnalysisClient(endpoint=document_ai_endpoint, credential=AzureKeyCredential(document_ai_key))


    def get_pdf_files_in_directory(self, directory_path):
        """
        The function `get_pdf_files_in_directory` returns a list of PDF files in a given directory.
        
        :param directory_path: The directory path is the path to the directory where you want to search
        for PDF files. It should be a string representing the absolute or relative path to the directory
        :return: a list of file paths for all the PDF files in the specified directory.
        """
        # Use a list comprehension to filter files with the ".pdf" extension
        pdf_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.lower().endswith((".pdf", ".PDF"))]
        return pdf_files


    def load_documents_into_langchain_local(self, document_files):
        """
        The function loads a list of document files into a local Langchain system for language analysis.
        
        :param document_files: The `document_files` parameter is a list of file paths to the documents
        that you want to load into the Langchain local environment. Each file path represents a document
        file that you want to process using the Langchain document intelligence capabilities
        :return: a list of langchain documents.
        """
        document_intelligence_loaders = []
    
        for document in document_files:
            document_intelligence_loader = DocumentIntelligenceLoader(file_path=document, client=self.document_analysis_client, model="prebuilt-document")
            document_intelligence_loaders.append(document_intelligence_loader)
    
        langchain_documents = []
    
        for document_intelligence_loader in document_intelligence_loaders:
            langchain_document = document_intelligence_loader.load()
            langchain_documents.append(langchain_document)
        
        flat_langchain_documents_local = [document for sublist in langchain_documents for document in sublist]

        return flat_langchain_documents_local


class BlobStorageProcessor:
    """ This Class is used to parse PDF documents present in the Azure Blob Storage Container
    """
    def __init__(self):
        """
        The function initializes the necessary clients for working with Azure Blob Storage and Document
        Analysis.
        """
        self.account_name = os.getenv("AZURE_BLOB_STORAGE_ACCOUNT_NAME")
        self.account_key = os.getenv("AZURE_BLOB_STORAGE_ACCOUNT_KEY")
        self.container_name = os.getenv("AZURE_BLOB_STORAGE_CONTAINER_NAME")
        self.blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_BLOB_STORAGE_CONNECTION_STRING"))
        self.container_client = self.blob_service_client.get_container_client(os.getenv("AZURE_BLOB_STORAGE_CONTAINER_NAME"))
        self.document_analysis_client = self.initialize_document_analysis_client()


    def initialize_document_analysis_client(self):
        """
        The function initializes a DocumentAnalysisClient object using the Azure Document Intelligence
        endpoint and key from environment variables.

        :return: a DocumentAnalysisClient object.
        """
        # Get the Azure Document Intelligence endpoint and key from environment variables
        document_ai_endpoint = os.getenv("AZURE_DOCUMENTAI_ENDPOINT")
        document_ai_key = os.getenv("AZURE_DOCUMENTAI_KEY")
        
        # Create and return a DocumentAnalysisClient object
        return DocumentAnalysisClient(endpoint=document_ai_endpoint, credential=AzureKeyCredential(document_ai_key))
    

    def generate_blob_url(self, blob_name):
        """
        The function to generate the blob URL (including the SAS token - with time limit) to 
        access the blobs from python

        :return: blob URL 
        """
        # Generating the SAS token
        blob_sas_token = generate_blob_sas(account_name=self.account_name, 
                                            container_name=self.container_name,
                                            blob_name=blob_name,
                                            account_key=self.account_key,
                                            permission=BlobSasPermissions(all=True),
                                            expiry=datetime.utcnow() + timedelta(hours=1))
        
        # Generating the URL - including the SAS token and one hour time limit
        blob_url = "https://"+self.account_name+".blob.core.windows.net/"+self.container_name+"/"+blob_name+"?"+blob_sas_token
        return blob_url


    def file_name_retain(self, document_pages, blob_name):
        """
        Function to retain the original file name in the metadata of the pages. Since, Azure Blob Storage uses 
        temp URL for files in the storage. 

        :return: pages with original file_name in metadata
        """
        
        # Inserting the original file_name in the metdata of all the pages in a document
        for page in document_pages:
            page.metadata['source'] = blob_name.split('/')[-1]
        return document_pages


    def process_pdfs_from_folder(self, folder_name):
        """
        The function `process_pdfs_from_blob` retrieves PDF files from a specified folder in a blob
        storage container, reads the content of each PDF file, and returns a list of loaded documents.
        
        :param folder_name: The `folder_name` parameter is the name of the folder where the PDF files
        are located
        :return: a list of langchain documents.
        """
        prefix = f"{folder_name}/"
        blob_list = self.container_client.list_blobs(name_starts_with=prefix)
        multiple_document_pages_collection = []
        
        # Loading all the PDF files (blobs) present in the folder
        for blob in blob_list:
            if blob.name.endswith((".pdf", ".PDF")):
                blob_url = self.generate_blob_url(blob.name)
                document_intelligence_loader = DocumentIntelligenceLoader(file_path=blob_url, client= self.document_analysis_client, model="prebuilt-read")
                single_document_pages = self.file_name_retain(document_intelligence_loader.load(), blob.name)
                multiple_document_pages_collection.append(single_document_pages)
        
        all_document_pages = [pages for document in multiple_document_pages_collection for pages in document]

        return all_document_pages


    def delete_folder(self, folder_name):
        """ Function to delete the folder in the Azure Blob Storage container.
        By deleting all the blobs present in the folder.

        :return: None object
        """
        for blob in self.container_client.list_blobs(name_starts_with=folder_name):
            self.container_client.delete_blob(blob.name)


def chunk_documents(documents, chunk_size, chunk_overlap):
    """
    The function `chunk_documents` takes a list of documents and splits them into smaller chunks using a
    text splitter.
    
    :param documents: The `documents` parameter is a list of strings, where each string represents a
    document that you want to split into smaller chunks
    :return: a list of chunks, which are the result of splitting the input documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks
