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

schema = Object(
    id="action",
    description="User is looking for sports tickets",
    attributes=[
        Text(
            id="sport",
            description="which sports do you want to buy tickets for?",
            examples=[
                (
                    "I want to buy tickets to basketball and football games",
                    ["basketball", "footbal"],
                )
            ],
        ),
        Text(
            id="location",
            description="where would you like to watch the game?",
            examples=[
                ("in boston", "boston"),
                ("in france or italy", ["france", "italy"]),
            ],
        ),
        Object(
            id="price_range",
            description="how much do you want to spend?",
            attributes=[],
            examples=[
                ("no more than $100", {"price_max": "100", "currency": "$"}),
                (
                    "between 50 and 100 dollars",
                    {"price_max": "100", "price_min": "50", "currency": "$"},
                ),
            ],
        ),
    ],
)

chain = create_extraction_chain(llm, schema, encoder_or_encoder_class="json")

output = chain.invoke(
    "I want to see a celtics game in boston somewhere between 20 and 40 dollars per ticket"
)["data"]

print(output)