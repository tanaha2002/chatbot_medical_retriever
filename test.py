import requests
import time
import streamlit as st
# URL where your FastAPI application is running
api_url = st.secrets['api_url']
def test_api():
    question = "địa chỉ vinmec hà nội"  
    response = requests.get(f"{api_url}/chat_engine_2", params={"question": question}, stream=True)
    if response.status_code == 200:
        for chunk in response.iter_content(chunk_size=None):
            text = chunk.decode("utf-8")
            if "https" in text:
                print(text)
            else:
                for char in text:
                    print(char, end="", flush=True)  
                    time.sleep(0.02)
    else:
        print(f"Failed with status code: {response.status_code}")
if __name__ == "__main__":
    test_api()