
import warnings
import streamlit as st
from utils.snowchat_ui import StreamlitUICallbackHandler, message_func, message_func_stream, create_messenge
import sys
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
import asyncio
import requests
import time


# Create a new event loop
loop = asyncio.new_event_loop()

# Set the event loop as the current event loop
asyncio.set_event_loop(loop)


INITIAL_MESSAGE = [
    {"role": "user", "content": "Chào!"},
    {
        "role": "assistant",
        "content": "Chào bạn, tôi là phiên bản demo của Chatbot Vinmec. Tôi có thể giúp gì cho bạn?",
    },
]
def reset_chat():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.session_state["messages"] = INITIAL_MESSAGE
    st.session_state["history"] = []


warnings.filterwarnings("ignore")
chat_history = []

st.title("Vinmec Retrieval")
st.caption("Hỏi đáp dựa trên nguồn dữ liệu của Vinmec")
options = [
    # ("✨ Chat Engine 1", "chat_engine"),
    ("♾️ DuckDuckGo Search", "chat_engine_2"),
    ("⛰️ Hybrid Engine", "hybrid_engine")
]
model = st.radio(
    "",
    options=[option for option in options],  # Display only the option texts
    index=1,
    format_func=lambda option: option[0],  # Display only the option text
    horizontal=True,
)





st.session_state["model"] = model

INITIAL_MESSAGE = [
    {"role": "user", "content": "Chào!"},
    {
        "role": "assistant",
        "content": "Chào bạn, tôi là phiên bản demo của Chatbot Vinmec. Tôi có thể giúp gì cho bạn?",
    },
]

with open("ui/sidebar.md", "r") as sidebar_file:
    sidebar_content = sidebar_file.read()

with open("ui/styles.md", "r") as styles_file:
    styles_content = styles_file.read()

# Display the DDL for the selected table
st.sidebar.markdown(sidebar_content)

# Create a sidebar with a dropdown menu




# Add a reset button

    
if st.sidebar.button("Reset Chat"):
    reset_chat()



st.write(styles_content, unsafe_allow_html=True)

# Initialize the chat messages history
if "messages" not in st.session_state.keys():
    st.session_state["messages"] = INITIAL_MESSAGE

if "history" not in st.session_state:
    st.session_state["history"] = []

if "model" not in st.session_state:
    st.session_state["model"] = model

# Prompt for user input and save
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    message_func(
        message["content"],
        True if message["role"] == "user" else False,
        True if message["role"] == "data" else False,
    )

callback_handler = StreamlitUICallbackHandler()




def append_chat_history(question, answer):
    st.session_state["history"].append((question, answer))





def append_message(content, role="assistant", display=False):
    message = {"role": role, "content": content}
    st.session_state.messages.append(message)
    if role != "data":
        append_chat_history(st.session_state.messages[-2]["content"], content)

    if callback_handler.has_streaming_ended:
        callback_handler.has_streaming_ended = False
        return


api_url=st.secrets['api_url']
def get_response(question, method):
    print(model)
    response = requests.get(
        f"{api_url}/{method}", 
        params={"question": question}, 
        stream=True
        )
    if response.status_code == 200:
        
        
        placeholder_source = st.empty()
        placeholder_text = st.empty()
        full_response = ""
        source = ""
        for chunk in response.iter_content(chunk_size=None):
            text = chunk.decode("utf-8")
            if "https" in text or "Tài liệu liên quan: \n" in text:
                source += text + "\n"
                placeholder_source = message_func_stream(source, placeholder_source,url = None)
            else:
                for char in text:
                    full_response += char
                    placeholder_text = message_func_stream(full_response, placeholder_text,url = None)
                    time.sleep(0.02)
        
        return full_response, source
    else:
        print(f"Failed with status code: {response.status_code}")
    

print(st.session_state["model"][1])

if st.session_state.messages[-1]["role"] != "assistant":
    content = st.session_state.messages[-1]["content"]
    if isinstance(content,str):

        response,source = get_response(content, st.session_state["model"][1])
        if source:
            append_message(source)
        append_message(response)
        
        
            
            
