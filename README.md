# ASz-Internship

This is a chatbot-application which uses RAG technology to make information from public patient brochures (from the Albert Schweitzer hospital in Dordrecht) more accesible. The main goal of this chatbot was to test the LLM's ability to give accesible and medically correct information. This was made for the internship I had in my 6th semester of my bachelor degree for HBO-ICT at the Christelijke Hogeschool Ede (CHE).

This chatbot was based on code from the following [source](https://medium.com/credera-engineering/build-a-simple-rag-chatbot-with-langchain-b96b233e1b2a). 

Before the application can be run locally the following needs to be configured or available:
- A python environment with the necessary libraries installed (can be found in requirements.txt).
- An .env file needs to be created with the API keys from your personal Pinecone and Hugging Face accounts (you may need to create these accounts if you don't already have those)

This application is purely made for testing things out, and since I ran into issues I have two chatbots. In the `streamlit.py` you can change the code to choose which chatbot you can use. The code you need to change is made clear with comments. The official bot still has issues and doens't implement the whole RAG process correctly (the context is still in vector format when the prompt is given to the LLM), so to still be able to test the chatbot I've made a hardcoded version which gives the text out of the pdf-file directly as context without embedding and searching. 

To start this application run the following line:
```
streamlit run streamlit.py
```

Found below is a screenshot of the application in action:
![image](https://github.com/mennojak/ASz-Internship/assets/84782061/8735d3a9-16bd-4739-a402-244c6ab23a33)

