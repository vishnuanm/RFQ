# Importing the required libraries
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self, path, is_folder=False, chunk_size=1000, chunk_overlap=200):
        """
        The function initializes an object with the given path, folder flag, chunk size, and chunk
        overlap values.
        
        :param path: The path parameter is a string that represents the file or folder path. It
        specifies the location of the file or folder that the code will be working with
        :param is_folder: The `is_folder` parameter is a boolean flag that indicates whether the `path`
        refers to a folder/directory or a file. If `is_folder` is set to `True`, it means that the
        `path` refers to a folder. If `is_folder` is set to `False, defaults to False (optional)
        :param chunk_size: The `chunk_size` parameter determines the size of each chunk when reading or
        processing data. It specifies the number of elements or bytes to be processed at a time. In this
        case, the chunk size is set to 1000, defaults to 1000 (optional)
        :param chunk_overlap: The `chunk_overlap` parameter determines the number of overlapping
        elements between consecutive chunks. In other words, when splitting a sequence into chunks, the
        `chunk_overlap` specifies how many elements from the previous chunk should be included in the
        next chunk, defaults to 200 (optional)
        """
        self.path = path
        self.is_folder = is_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def load_documents(self):
        """
        The function `load_documents` loads documents from a folder or a single file using the
        PyPDFLoader or PyPDFDirectoryLoader class, respectively.

        :return: The `load_documents` function returns the `documents` variable, which is the result of
        calling the `load()` method on the `loader` object.
        """
        if self.is_folder:
            loader = PyPDFDirectoryLoader(self.path)
        else:
            loader = PyPDFLoader(self.path)
        
        documents = loader.load()
        return documents


    def split_documents(self, documents):
        """
        The function splits a list of documents into smaller chunks using a recursive character text
        splitter.
        
        :param documents: The "documents" parameter is a list of strings, where each string represents a
        document that you want to split into smaller chunks
        :return: the chunks of text that result from splitting the input documents.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        return chunks


    def process(self):
        """
        The function "process" loads documents and splits them into chunks.

        :return: the chunks of documents after they have been processed.
        """
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        return chunks

# if __name__ == "__main__":
#     path = "rfq/"
#     processor = PDFProcessor(path)
#     chunks = processor.process()

#     for chunk in chunks:
#         # Process each chunk
#         print(chunk)
