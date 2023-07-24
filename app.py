import os
from apikey import apikey
from langchain.llms import HuggingFaceHub
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

#App Framework
os.environ['HUGGINGFACEHUB_API_TOKEN'] = apikey
st.title("WhatsTheMove AI")
prompt = st.text_input('Tell us where you are')

#Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'what are the top 3 tourist attractions in {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template = 'Summarize this wikipedia research {wikipedia_research} on {title} in 5 sentences'
)


#Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


#LLMs
repo_id = "google/flan-t5-xxl"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 1, "verbose":True, "max_length": 300})
title_chain = LLMChain(llm=llm, prompt = title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt = script_template, verbose=True, output_key='script', memory=script_memory)
#sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'],verbose=True)
wiki = WikipediaAPIWrapper()

#Show Response given prompt
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script=script_chain.run(title=title, wikipedia_research=wiki_research)

    #response = sequential_chain({'topic':prompt})
    st.write(title)
    st.write(script)

    with st.expander('City History'):
        st.info(title_memory.buffer)

    with st.expander('Information about Attractions'):
        st.info(script_memory.buffer)
    
    with st.expander('More Research'):
        st.info(wiki_research)



