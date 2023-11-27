# Import the required libraries
import io
import os
import pandas as pd
import datetime
import sys
import time
sys.path.append(".")
from dotenv import load_dotenv
load_dotenv()

# Importing Streamlit libraries
from streamlit_lottie import st_lottie
from PIL import Image
import streamlit as st

# Importing the custom module and Azure modules
from rfq_extraction.main import RFQExtractionAzure
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError
from rfq_extraction.parse import RichExcelWriter, create_excel_with_formatting


# # Loading animation and images
# st.lottie = st_lottie
# image = Image.open('Images/logo.png')
# st.sidebar.image(image, width=200)


def main():
    # Name and Path Creation
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    folder_name = f"rfq_{timestamp}"
    search_index_name = f"rfq-search-index-{timestamp}"
    output_file_path =  os.path.join(folder_name, "output.xlsx")

    # Azure Blob Storage Connection - initialising the client
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_BLOB_STORAGE_CONNECTION_STRING"))
    container_name = os.getenv("AZURE_BLOB_STORAGE_CONTAINER_NAME")
    container_client = blob_service_client.get_container_client(os.getenv("AZURE_BLOB_STORAGE_CONTAINER_NAME"))

    # Allow the user to upload multiple PDF files from the sidebar
    uploaded_pdf_files = st.file_uploader("Upload your RFQ files here", type=["pdf"], accept_multiple_files=True)

    # Adding a "Submit" button to the sidebar
    submit_button = st.button("Submit")

    # Upload the PDF files to timestamp generated folder name to the Azure Blob Storage container
    if submit_button:
        start_time = time.time()
        if uploaded_pdf_files:
            with st.spinner("Uploading..."):
                for uploaded_file in uploaded_pdf_files:
                    blob_name = f"{folder_name}/{uploaded_file.name}"
                    blob_client = container_client.get_blob_client(blob_name)
                    blob_client.upload_blob(uploaded_file, overwrite = True)
            st.success(f"{len(uploaded_pdf_files)} files uploaded to Azure Blob Storage!")

            # Calling the RFQ function in the main file to process the PDF files uploaded to the folder
            with st.spinner("Processing the RFQ..."):
                rfq_extraction_obj = RFQExtractionAzure(folder_name, search_index_name, output_file_path)
                result_df = rfq_extraction_obj.rfq_extraction()
                st.write("RFQ Processed Successfully!")
                import pickle
                pickle.dump(result_df, open('./result_df.p', 'wb'))

            # Creating a download link for the output Excel file
            with st.spinner("Generating the Excel Output..."):
                output = create_excel_with_formatting(result_df, 'rfq_output.xlsx', 'output')
                with open('rfq_output.xlsx', 'rb') as f:
                        file_data = f.read()
                st.write("Done! Click below to download the Excel output.")
            
            # Calculating the time taken to run the script
            end_time = time.time()
            st.write(f"Time Taken: {end_time-start_time} seconds")

            # Generating a link to download the Excel file
            st.download_button(
                label="Download Excel File",
                data=file_data,
                key="download_excel_button",
                file_name="rfq_output.xlsx"
            )

# if __name__=="__main__":
#     main()