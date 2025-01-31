import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

groq_api = os.getenv('GROQ_API')
model = "llama-3.2-90b-vision-preview"

# Define the prompt template
template = """
    Below is a draft text that may be poorly worded.
    Your goal is to:
    - Properly redact the draft text
    - Convert the draft text to a specified tone
    - Convert the draft text to a specified dialect

    Here are some examples of different Tones:
    - Formal: Greetings! OpenAI has announced that Sam Altman is rejoining the company as its Chief Executive Officer. After a period of five days of conversations, discussions, and deliberations, the decision to bring back Altman, who had been previously dismissed, has been made. We are delighted to welcome Sam back to OpenAI.
    - Informal: Hey everyone, it's been a wild week! We've got some exciting news to share - Sam Altman is back at OpenAI, taking up the role of chief executive. After a bunch of intense talks, debates, and convincing, Altman is making his triumphant return to the AI startup he co-founded.  

    Here are some examples of words in different dialects:
    - American: French Fries, cotton candy, apartment, garbage, cookie, green thumb, parking lot, pants, windshield
    - British: chips, candyfloss, flag, rubbish, biscuit, green fingers, car park, trousers, windscreen

    Example Sentences from each dialect:
    - American: Greetings! OpenAI has announced that Sam Altman is rejoining the company as its Chief Executive Officer.
    - British: On Wednesday, OpenAI, the esteemed artificial intelligence start-up, announced that Sam Altman would be returning as its Chief Executive Officer.

    Please start the redaction with a warm introduction. Add the introduction if you need to.
    
    if dialect is hindi then give me text into hindi language
    if dialect is marathi then give me text into marathi language
    Below is the draft text, tone, and dialect:
    DRAFT: {draft}
    TONE: {tone}
    DIALECT: {dialect}

    YOUR {dialect} RESPONSE:
"""

prompt = PromptTemplate(
    input_variables=["tone", "dialect", "draft"],
    template=template,
)

# Function to load the language model
def load_llm():
    return ChatGroq(temperature=0.6, api_key=groq_api, model=model)

# Streamlit UI Setup
st.set_page_config(page_title="Re-write Your Text", layout="centered")
st.header("Re-write Your Text with AI ✍️")
st.markdown("Enhance your writing by selecting a tone and dialect!")

# User input for text
draft_text = st.text_area(
    label="Enter your text:",
    placeholder="Type or paste your text here...",
    height=200
)

if len(draft_text.split()) > 700:
    st.error("⚠️ Please enter a shorter text. The maximum length is 700 words.")
    st.stop()

# Tone and dialect selection
col1, col2 = st.columns(2)

with col1:
    tone_option = st.selectbox("Choose a tone:", ["Formal", "Informal"], index=0)

with col2:
    dialect_option = st.selectbox("Choose an English dialect:", ["American", "British" ,"hindi","marathi"], index=0)

# Generate button
if st.button("Generate ✨"):
    if draft_text:
        with st.spinner("Rewriting your text... ⏳"):
            llm = load_llm()
            formatted_prompt = prompt.format(
                tone=tone_option,
                dialect=dialect_option,
                draft=draft_text
            )
            improved_text = llm.invoke(formatted_prompt)  # Corrected function call
        
        st.success("✅ Text successfully rewritten!")
        st.write(improved_text.content)
