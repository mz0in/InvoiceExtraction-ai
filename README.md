# InvoiceExtraction
Extracting Invoice data using ChatGPT, LangChain and Kor

This project demonstrates the extraction of relevant information from invoices using the GPT-3.5 language model. It utilizes the kor.extraction module and the langchain.chat_models module for creating extraction chains and interacting with the GPT-3.5 model, respectively.

## Installation

To run this project, please follow these steps:

1. Clone the repository: git clone https://github.com/your-username/InvoiceExtraction.git
2. Install the required dependencies: pip install -r requirements.txt
3. Set up the API credentials:
   - Obtain an API key for GPT-3.5 from OpenAI.
   - Create a file named creds.py in the project root directory.
   - Inside creds.py, define a variable gpt_api_key and assign your GPT-3.5 API key to it.
4. Save the file.

## Usage
1. Import the necessary modules and credentials:

```
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
from langchain.chat_models import ChatOpenAI

# Load GPT 3.5 model
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    max_tokens=2000
)
```

2. Define the schemas for extracting invoice information:

- invoice_number_schema: Extracts the unique identifier of the invoice.
- invoice_date_schema: Extracts the date from the invoice.
- address_schema: Extracts the address details from the invoice.
- billing_address_schema: Extracts the billing address from the invoice.
- products_schema: Extracts the product details from the invoice.
- total_bill_schema: Extracts the total bill information from the invoice.
- invoice_schema: Combines all the above schemas to extract relevant invoice information.

```
invoice_number_schema = Object(
    id="invoice_number_extraction",
    description="extraction of relevant information from invoice",
    attributes=[
        Text(
            id="invoice_number",
            description="unique number (identifier) of given invoice",
            examples=[
                ("Invoice Number: INV-23490", "INV-23490"),
                ("INVNO-76890", "INVNO-76890"),
                ("Invoice: INV-100021", "INV-100021")
            ],
        )
    ],
    many=False,
)
```

Define other schemas similarly

```
invoice_schema = Object(
    id="invoice_information",
    description="relevant invoice parsing from raw extracted text",
    attributes=[
        invoice_number_schema,
        invoice_date_schema,
        billing_address_schema,
        products_schema,
        total_bill_schema,
    ],
    many=True,
)
```

Create the extraction chain and process the invoice text:

```
invoice_chain = create_extraction_chain(llm, invoice_schema, encoder_or_encoder_class="json")

processed_text = '''SAMPLE INVOICE GPT Solutions 123 Marvel Street Los Angeles, CA, 90007 ...'''

invoice_data = invoice_chain.predict_and_parse(text=processed_text)['data']
```

Use the extracted invoice data as per your requirements.

## Contributing
Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgments
- The GPT-3.5 language model is developed by OpenAI. Visit their website for more information.
- The kor library provides useful tools for text extraction. Visit their GitHub repository for documentation and examples.
- The langchain library simplifies interactions with GPT-based models. Visit their GitHub repository for more details.
