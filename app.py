import streamlit as st
from streamlit_chat import message
import os

from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationEntityMemory
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Qdrant
import qdrant_client

st.set_page_config(page_title="ChatDiary", page_icon="📖")
st.header("📖 ChatDiary")

is_gpt4 = st.checkbox('Enable GPT4',help="With this it might get slower")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_chat_history(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAI:{ai}")
    return "\n".join(res)

now = datetime.now()

_DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context.
AI wants to help reflect the day. If anything relevant in the past is helpful to show empathy, then use.
the conversation needs to be in Japanese. Only return AI's part, don't act as Human.
Relevant pieces of previous conversation:
{history}

(You do not need to use these pieces of information if not relevant)

Current conversation:
Human: {input}
AI:"""
PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
)

if is_gpt4:
    model = "gpt-4"
else:
    model = "gpt-3.5-turbo"
    
llm = ChatOpenAI(temperature=0.9, model_name=model, streaming=True, verbose=True)
embeddings = OpenAIEmbeddings()
client = qdrant_client.QdrantClient(url=os.environ['QDRANT_URL'], prefer_grpc=True, api_key=os.environ['QDRANT_API_KEY'])
db = Qdrant(client=client, collection_name="yasuhiro", embeddings=embeddings)
retriever = db.as_retriever(search_kwargs=dict(k=1))
memory = ConversationEntityMemory(retriever=retriever)

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    chain = ConversationChain(
        llm=llm,
        prompt=PROMPT,
        memory=memory,
        verbose=True
    )
    return chain
    
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

with st.form(key='chat'):
    user_input = get_text()
    chat_button = st.form_submit_button('💬')

if chat_button:
    with st.spinner('typing...'):
        chat_history = []
        chain = load_chain()
        memory_variables = memory.load_memory_variables({"input_key": user_input})
        st.write(memory_variables)
        result = chain.predict(input= str(now) + ": " + user_input)
        memory.save_context({"input": user_input, "now": now}, {"output": result})
        
        st.session_state.past.append(user_input)
        st.session_state.generated.append(result)


if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        try:
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        except:
            pass
