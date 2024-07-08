## Installation

 First install the necessary libraries: 

```bash
pip install -r requirements.txt
```

Next create .env file with necessary api keys.

```bash
OPENAI_API_KEY="**********"
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="**********"
```

Next run server.py

```bash
python server.py   
```

Next run app.py

```bash
streamlit run app.py
```
