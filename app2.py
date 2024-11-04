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
    id="person",
    description="Personal information",
    examples=[
        ("Alice and Bob are friends", [{"first_name": "Alice"}, {"first_name": "Bob"}])
    ],
    attributes=[
        Text(
            id="first_name",
            description="The first name of a person.",
        )
    ],
    many=True,
)

chain = create_extraction_chain(llm, schema)

output = chain.invoke(("My name is Bobby. My brother's name Joe."))["data"]

print(output)

"""
messages = [
    {"role": "user", "content": "Tell me a joke."}
]

messages = [
    SystemMessage(content="Translate the following from English into Chinese"),
    HumanMessage(content="latest news"),
]

llm.invoke(messages)


response = llm.invoke(messages)
# print(response["choices"][0]["message"]["content"])
print(response.content)
"""