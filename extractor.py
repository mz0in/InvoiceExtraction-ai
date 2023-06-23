from kor.extraction import create_extraction_chain  # Import the create_extraction_chain function
from kor.nodes import Object, Text, Number  # Import necessary classes from kor.nodes
from langchain.chat_models import ChatOpenAI  # Import the ChatOpenAI model

# Load the GPT 3.5 model
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    max_tokens=2000
)

# Define the schema for extracting the invoice number
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
            ]
        )
    ],
    many=False
)

# Define the schema for extracting the invoice date
invoice_date_schema = Object(
    id="date_extraction",
    description="extraction of date from the invoice",
    attributes=[
        Text(
            id="date",
            description="invoice date",
            examples=[
                ("Date: 05/14/23", "05/14/23"),
                ("14/5/23", "14/05/23"),
                ("Invoice: 14/05/2023", "14/5/2023")
            ]
        )
    ],
    many=False
)

# Create the extraction chain for invoice date
date_chain = create_extraction_chain(llm, invoice_date_schema)

# Define the schema for extracting address details
address_schema = Object(
    id="address",
    description="address details",
    attributes=[
        Text(id="name", description="the name of person and organization"),
        Text(id="address_line", description="the local delivery information such as street, building number, PO box, or apartment portion of a postal address"),
        Text(id="city", description="the city portion of the address"),
        Text(id="state_province_code", description="the code for address US states"),
        Number(id="postal_code", description="the postal code portion of the address")
    ],
    examples=[
        (
            "James Bond, Bond Industries 5000 Forbes Avenue Pittsburgh, PA 15213",
            {
                "name": "James Bond, Bond Industries",
                "address_line": "Bond Industries 5000 Forbes Avenue",
                "city": "Pittsburgh",
                "state_province_code": "PA",
                "postal_code": "15213",
            },
        ),
        (
            "Kaushik Shakkari 840 Childs Way, Los Angeles, CA 90089",
            {
                "name": "Kaushik Shakkari",
                "address_line": "840 Childs Way",
                "city": "Los Angeles",
                "state_province_code": "CA",
                "postal_code": "90089",
            },
        ),
        (
            "Shakkari Solutions PO Box 1234 Atlanta GA 30033",
            {
                "name": "Shakkari Solutions",
                "address_line": "PO Box 1234",
                "city": "Atlanta",
                "state_province_code": "GA",
                "postal_code": "30033",
            },
        )
    ],
    many=True
)

# Define the billing address schema based on the address schema
billing_address_schema = address_schema.replace(
    id="billing_address",
    description="where the bill for a product or service is sent so it can be paid by the recipient"
)

# Define the schema for extracting product details from the invoice
products_schema = Object(
    id="bill",
    description="the details of bill",
    attributes=[
        Text(id="product_description", description="the description of the product or service"),
        Text(id="count", description="number of units bought for the product"),
        Text(id="unit_item_price", description="price per unit"),
        Text(id="product_total_price", description="the total price, which is number of units * unit_price"),
    ],
    examples=[
        (
            "iphone 14 pro black 2 $1200.00 $2400.00",
            {
                "product_description": "iphone 14 pro black",
                "count": 2,
                "unit_item_price": 1200,
                "product_total_price": 2400,
            },
        ),
    ],
    many=True
)

# Define the schema for extracting total bill information
total_bill_schema = Object(
    id="total_bill",
    description="the details of total amount, discounts and tax",
    attributes=[
        Number(id="total", description="the total amount before tax and delivery charges"),
        Number(id="discount_amount", description="discount amount is total cost * discount %"),
        Number(id="tax_amount", description="tax amount is tax_percentage * (total - discount_amount). If discount_amount is 0, then its tax_percentage * total"),
        Number(id="delivery_charges", description="the cost of shipping products"),
        Number(id="final_total", description="the total price or balance after removing tax, adding delivery and tax from total"),
    ],
    examples=[
        (
            "total $100000.00 discount 0% tax 5 percentage delivery cost $100.00 final_total $95100.00",
            {
                "total": 100000,
                "discount_amount": 0,
                "tax_amount": 5000,
                "delivery_charges": 100,
                "final_total": 105100
            },
        ),
        (
            "Amount Now Due: $250,000",
            {
                "total": 250000,
                "discount_amount": 0,
                "tax_amount": 0,
                "delivery_charges": 0,
                "final_total": 250000
            },
        ),
    ],
    many=False
)

# Define the invoice schema combining all the extracted information
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
    many=True
)

# Create the extraction chain for the invoice schema
invoice_chain = create_extraction_chain(llm, invoice_schema, encoder_or_encoder_class="json")

# Processed text containing the invoice information
processed_text = '''<ENTER THE TEXT>'''

# Predict and parse the invoice information from the processed text using the extraction chain
invoice_data = invoice_chain.predict_and_parse(text=processed_text)['data']
