from unstructured_client import UnstructuredClient
from unstructured_client.models import operations, shared
import PyPDF2
import os
import base64
from PIL import Image
import io
from docx2pdf import convert
from config import api_key_auth,server_url

def process_pdf(path):

    if path.endswith(".docx"):
        # Convert .docx to PDF (using the same directory)
        pdf_path = path.replace(".docx", ".pdf")
        convert(path, pdf_path)
        path = pdf_path  # Update path to the new PDF
    
    client = UnstructuredClient(
            api_key_auth=api_key_auth,
            server_url=server_url,
    )

    # Source: https://github.com/Unstructured-IO/unstructured/blob/main/example-docs/embedded-images-tables.pdf
    
    # Where to get the input file and store the processed data, relative to this .py file.
    local_input_filepath = path
    #local_output_filepath = "D:\RAG\documents\embedded-images-tables.json"

    with open(local_input_filepath, "rb") as f:
        files = shared.Files(
            content=f.read(),
            file_name=local_input_filepath
        )

    request = operations.PartitionRequest(
        shared.PartitionParameters(
            files=files,
            strategy=shared.Strategy.HI_RES,
            split_pdf_page=True,
            split_pdf_allow_failed=True,
            split_pdf_concurrency_level=15,
            # Extract the Base64-encoded representation of each
            # processed "Image" and "Table" element. Extract each into
            # an "image_base64" object, as a child of the
            # "metadata" object, for that element in the result.
            # Element type names, such as "Image" and "Table" here,
            # are case-insensitive.
            # Any available Unstructured element type is allowed.
            extract_image_block_types=["Image", "Table"]
        )
    )
    save_directory = "documents/"
    try:
        result = client.general.partition(request)
        image_count = 0
        table_count = 0

        for element in result.elements:
            if "image_base64" in element["metadata"]:
                # Decode the Base64-encoded representation of the 
                # processed "Image" or "Table" element into its original
                # visual representation, and then show it.
                image_data = base64.b64decode(element["metadata"]["image_base64"])
                image = Image.open(io.BytesIO(image_data))
                element_type = element.get("type", "").lower()

                page_number = element.get("metadata", {}).get("page_number", 1)

            if element_type == "image":
                image_count += 1
                image.save(os.path.join(save_directory, f"image_page{page_number}_{image_count}.png"))
                print(f"Image {image_count} on page {page_number} saved as image_page{page_number}_{image_count}.png")
                
            elif element_type == "table":
                table_count += 1
                image.save(os.path.join(save_directory, f"table_page{page_number}_{table_count}.png"))
                print(f"Table {table_count} on page {page_number} saved as table_page{page_number}_{table_count}.png")

    except Exception as e:
        print(e)

    files = open(local_input_filepath, 'rb')
    pdfReader = PyPDF2.PdfReader(files)

    for i, pageObj in enumerate(pdfReader.pages):
        page_text = pageObj.extract_text()
    #  print(page_text)
        
        # Build the file path correctly
        file_path = os.path.join(save_directory, f"text_page{i+1}.txt")
        
        # Open the output file in append mode
        with open(file_path, 'a',encoding='utf-8') as out_file:
            out_file.write(page_text)

    files.close()
