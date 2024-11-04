import os
os.environ["OPENAI_API_BASE"] = "http://localhost:11434"

from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(
    api_key="ollama",
    model="llama3",
    base_url="http://localhost:11434/v1/",
    temperature=0,
    max_tokens=2000,
)

from_address = Object(
    id="from_address",
    description="Person moved away from this address",
    attributes=[
        Text(id="street"),
        Text(id="city"),
        Text(id="state"),
        Text(id="zipcode"),
        Text(id="country", description="A country in the world; e.g., France."),
    ],
    examples=[
        (
            "100 Main St, Boston, MA, 23232, USA",
            {
                "street": "100 Marlo St",
                "city": "Boston",
                "state": "MA",
                "zipcode": "23232",
                "country": "USA",
            },
        )
    ],
)

to_address = from_address.replace(
    id="to_address", description="Address to which the person is moving"
)

schema = Object(
    id="information",
    attributes=[
        Text(
            id="person_name",
            description="The full name of the person or partial name",
            examples=[("John Smith was here", "John Smith")],
        ),
        from_address,
        to_address,
    ],
    many=True,
)

chain = create_extraction_chain(
    llm, schema, encoder_or_encoder_class="json", input_formatter=None
)

"""
output = chain.invoke(
    "Alice Doe moved from New York to Boston, MA while Bob Smith did the opposite."
)["data"]
"""


output = chain.invoke(
    "Alice Doe and Bob Smith moved from New York to Boston. Andrew was 12 years"
    " old. He also moved to Boston. So did Joana and Paul. Betty did the opposite."
)["data"]

print(output)