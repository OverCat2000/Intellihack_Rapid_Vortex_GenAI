# HappyHire: Generative AI-Based Recruitment Management System

HappyHire is a sophisticated recruitment management system powered by Generative AI. It automates several key tasks in the recruitment process, enhancing efficiency and candidate experience.

## Features

Resume Scoring: Automatically scores resumes based on comparison with job descriptions.
Candidate Ranking: Identifies top candidates based on scored resumes.
Personalized Feedback: Sends personalized feedback to rejected candidates on resume improvement ideas.
Interview Management: Sends interview invitations to selected candidates.
Personalized Thank You Images: Generates personalized thank you images reflecting the candidate's resume summary.

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

Next run main file inside email directory to run email services
