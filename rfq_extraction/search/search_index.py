# Importing the required libraries
import os
from dotenv import load_dotenv
load_dotenv()
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchField,
    SearchableField,
    SearchFieldDataType, 
    VectorSearch,
    HnswVectorSearchAlgorithmConfiguration,
    HnswParameters,
    VectorSearchProfile,
    SemanticConfiguration,
    PrioritizedFields,
    SemanticField,
    SemanticSettings,
)


class PDFSemanticSearch:
    def __init__(self,search_index_name, use_semantic_configuration=False):
        """
        The function initializes an object with the given search index name, Azure search endpoint,
        Azure search key, and a flag indicating whether to use semantic configuration.
        
        :param search_index_name: The search_index_name parameter is the name of the search index that
        you want to use for your Azure Search service. This is the index where your data will be stored
        and indexed for searching

        :param use_semantic_configuration: The `use_semantic_configuration` parameter is a boolean value
        that determines whether to use a semantic configuration for the search index. If set to `True`,
        a semantic configuration will be used. If set to `False`, a default configuration will be used,
        defaults to False (optional)
        """
        self.search_index_name = search_index_name
        self.azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.azure_search_key = os.getenv("AZURE_SEARCH_KEY")
        self.use_semantic_configuration = use_semantic_configuration


    def create_search_index(self):
        """
        The function `create_search_index` creates a search index with specified fields and Vector
        search and Semantic search configuration in Azure Cogitive Search.
        
        :param self: Refer to the current object
        :return: The index that was created
        """
        fields = [
                SimpleField(name="documentId", type=SearchFieldDataType.String, filterable=True, sortable=True, key=True),     
                SearchableField(name="content", type=SearchFieldDataType.String),
                SearchableField(name="source_document_name", type=SearchFieldDataType.String), 
                SearchField(name="page_number", type=SearchFieldDataType.Int64, sortable=True),
                SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), 
                searchable=True, vector_search_dimensions = 1536, vector_search_profile="myHnswProfile")
                ]
                
        vector_search = VectorSearch(
            algorithms=[
                HnswVectorSearchAlgorithmConfiguration(
                    name="myHnsw",
                    kind="hnsw",
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric="cosine",
                    )
                ),
            ],
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm="myHnsw",
                ),
            ],
        )

        if self.use_semantic_configuration:
            semantic_config = SemanticConfiguration(
                name="my-semantic-config",
                prioritized_fields=PrioritizedFields(
                    prioritized_content_fields=[
                        SemanticField(field_name="content")]
                )
            )

            semantic_settings = SemanticSettings(configurations=[semantic_config])
        else:
            semantic_settings = None

        client = SearchIndexClient(self.azure_search_endpoint, AzureKeyCredential(self.azure_search_key))

        index = SearchIndex(
            name=self.search_index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_settings=semantic_settings
        )
        client.create_index(index)


    def delete_search_index(self):
        """
        The delete_search_index function deletes the search index created in the Azure Cognitive Search.
        
        :param self: Represent the instance of the class
        :return: A none value
        """
        try:
            client = SearchIndexClient(self.azure_search_endpoint, AzureKeyCredential(self.azure_search_key))
            client.delete_index(self.search_index_name)
        except Exception as e:
            print(str(e))

