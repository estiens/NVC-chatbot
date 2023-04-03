import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from PIL import Image

prompt_template = """You are an expert in NVC (non-violent communication).
Your role is a coach to help people communicate more empathically.
You can help them rephrase what they want to say using NVC principles,
and you can also help them figure out how to respond to someone else.
You should be friendly, non-judgemental and encouraging.
You can also use some humor if it is appropriate.
----------------
{question}"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["question"]
)

def load_chain():
  llm = ChatOpenAI(model_name ="gpt-4",verbose=True,temperature=0.7,max_tokens=8000)
  chain = LLMChain(llm=llm, prompt=PROMPT)
  return chain

if "chain" not in st.session_state:
    chain = load_chain()
    st.session_state["chain"] = chain

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

if input not in st.session_state:
    st.session_state["input"] = ""

# From here down is all the StreamLit UI.
st.set_page_config(page_title="NVC demo", page_icon=":robot:")

with st.sidebar:
  col1, col2 = st.columns(2)
  with col1:
    st.markdown("**NVC AI MentorBot**")
    st.markdown("(Name: Nambo)")

  with col2:
    icon = Image.open('icon.png')
    st.image(icon, use_column_width=True)
  st.markdown(
    """This mini-app helps you practice NVC (non-violent communication) by
      asking an AI mentor about situations you may be in, things you want to rephrase,
      or things you want to say to someone else. The AI will help you rephrase your
      statements, response empathetically, or just re-think a situation"""
  )

def get_response(user_input):
    with st.spinner("Thinking about that"):
      output = st.session_state.chain.run(question=user_input)
    return output

# def get_text():
#     input_text = st.text_input(label="What would you like to rephrase or think through?",key="input", on_change=clear_text)
#     return input_text

if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

def clear_text():
    st.session_state["user_input"] = st.session_state["input"]
    st.session_state["input"] = ""

# input = st.text_input("Input window", key="input", )
st.text_input(label="What would you like to rephrase or think through? You can try something like 'please rephrase you always ask me so many questions, I feel like I'm being interrogated' or perhaps 'My partner is angry at me for not washing the dishes but I just forgot!' or 'My boss just insulted me for my work quality' or even just ask about a general situation.",key="input", on_change=clear_text)

if st.session_state["user_input"]:
    user_input = st.session_state["user_input"]
    st.write("You: ", user_input)
    output = get_response(user_input)
    st.write("Nambo: ", output)
    # st.text_area(label="Nambo says...", value=output, height=300)
    st.session_state.generated.append(output)
    st.session_state.past.append(user_input)

# if st.session_state["generated"]:
#   for i in range(len(st.session_state["generated"]) - 1, -1, -1):
#     with st.expander("Past conversations"):
#       c.write(st.session_state["past"][i])
#       c.write(st.session_state["generated"][i])
