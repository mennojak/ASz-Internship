from chatbot import ChatBot
from chatbot_hardcoded import ChatBot_Harcoded
import re
import streamlit as st

# The two options depending on which chatbot you use, the harcoded one or the one with RAG (which currently doens't implement it well)
# Comment out the one you don't use.
#
#
# bot = ChatBot()
bot = ChatBot_Harcoded()
#   
#

st.set_page_config(page_title="ASz Folder assistent")
with st.sidebar:
    st.title('ASz Folder assistent')

def generate_response(input):
    
    # The two options depending on which chatbot you use, the hardcoded one uses the ask_question function, the others use rag_chain.invoke
    # Comment out the one you don't use.
    #   
    #
    # result = bot.rag_chain.invoke(input)
    result = bot.ask_question(input)
    #   
    #

    answer_start = result.rfind("Antwoord:") + len("Antwoord:")
    answer = result[answer_start:].strip()
    return answer


if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hallo, waar kan ik u vandaag mee helpen?"}]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Bezig met het opstellen van een antwoord voor u.."):
            response = generate_response(input) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)