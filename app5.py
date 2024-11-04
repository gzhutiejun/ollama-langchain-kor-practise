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
    id="player",
    description=(
        "User is controlling a music player to select songs, pause or start them or play"
        " music by a particular artist."
    ),
    attributes=[
        Text(
            id="song",
            description="User wants to play this song",
            examples=[],
            many=True,
        ),
        Text(
            id="album",
            description="User wants to play this album",
            examples=[],
            many=True,
        ),
        Text(
            id="artist",
            description="Music by the given artist",
            examples=[("Songs by paul simon", "paul simon")],
            many=True,
        ),
        Text(
            id="action",
            description="Action to take one of: `play`, `stop`, `next`, `previous`.",
            examples=[
                ("Please stop the music", "stop"),
                ("play something", "play"),
                ("play a song", "play"),
                ("next song", "next"),
            ],
        ),
    ],
    many=False,
)

chain = create_extraction_chain(llm, schema, encoder_or_encoder_class="json")

#output = chain.invoke("stop playing the music now")['data']
#output = chain.invoke("i want to hear a song")["data"]
#output = chain.invoke("can you play the lion king soundtrack")["data"]
output = chain.invoke("play songs by paul simon and led zeppelin and the doors")
print(output)