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
    id="personal_info",
    description="Personal information about a given person.",
    attributes=[
        Text(
            id="first_name",
            description="The first name of the person",
            examples=[("John Smith went to the store", "John")],
        ),
        Text(
            id="last_name",
            description="The last name of the person",
            examples=[("John Smith went to the store", "Smith")],
        ),
        Number(
            id="age",
            description="The age of the person in years.",
            examples=[("23 years old", "23"), ("I turned three on sunday", "3")],
        ),
    ],
    examples=[
        (
            "John Smith was 23 years old. He was very tall. He knew Jane Doe. She was 5 years old.",
            [
                {"first_name": "John", "last_name": "Smith", "age": 23},
                {"first_name": "Jane", "last_name": "Doe", "age": 5},
            ],
        )
    ],
    many=True,
)


chain = create_extraction_chain(llm, schema)
print(chain.get_prompts()[0].format_prompt(text="[user input]").to_string())