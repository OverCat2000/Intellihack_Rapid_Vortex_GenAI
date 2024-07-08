from flask import Flask, jsonify, request
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langgraph.graph import END, StateGraph
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.prompts import PromptTemplate
import json
import pandas as pd
from typing_extensions import TypedDict
from typing import List
import openai
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import requests
from PIL import Image
from io import BytesIO
import os
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

app = Flask(__name__)


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
client = openai.OpenAI(api_key="sk-proj-wjzCVdqCphIO3KIUhyoMT3BlbkFJEKI2Z4S4VzZ4EWJb5I1A")



extract_personal_info_prompt = PromptTemplate(
        template="""Extract the following personal information from the resume provided below. 
        If any information is not available, please state "Not available" for that field. 
        The information should be outputted in JSON format.

        RESUME CONTENT:\n\n {resume} \n\n

        Expected JSON keys and values:
        
            "name": "Full name or 'Not available'",
            "email": "Email address or 'Not available'",
            "contact_number": "Contact number or 'Not available'",
            "linkedIn": "LinkedIn profile URL or 'Not available'"
        
        """,
        input_variables=["resume"],
    )

extracted_personal_info_generator = extract_personal_info_prompt | llm | JsonOutputParser()



resume_scorerer_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a Talent Acquisition agent. 
    You are an expert at identifying spelling and grammar mistakes, educational background and certifications, work experience and projects, technical skills and soft skills mentioned in applicant resumes.

     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Conduct a thorough analysis of the provided resume against the given job description. Evaluate the resume and assign scores accordingly. The maximum score is 675 points, and the minimum score is 0.

    Only give final score. No explanation needed!
    Expected JSON keys and values: "resume_score" : "score applicant get"
    
    RESUME CONTENT:\n\n {resume} \n\n
    JOB DESCRIPTION:\n\n {job} \n\n
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["resume", "job"],
)

resume_score_generator= resume_scorerer_prompt | llm | JsonOutputParser()



def write_personal_info_file(name, email, contact_number, linkedIn, filename = "personal_info.csv"):
  """Writes the given content as a markdown file to the local directory.

  Args:
    content: The string content to write to the file.
    filename: The filename to save the file as.
  """
  try:
    with open(filename, 'a') as file:
      file.write(f"{name},{email},{contact_number},{linkedIn}\n")
  except IOError as e:
    print(f"Error writing to file: {e}")
  except Exception as e:
    print(f"Unexpected error: {e}")





### State

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        initial_email: email
        email_category: email category
        draft_email: LLM generation
        final_email: LLM generation
        research_info: list of documents
        info_needed: whether to add search info
        num_steps: number of steps
    """
    applicant_resume_path: str
    applicant_resume: List[str]
    job_description: List[str]
    applicant_name : str
    applicant_email : str
    

def extract_personal_info(state):

    print("---EXTRACT PERSONAL INFO---")

    applicant_resume_path = state['applicant_resume_path']
    applicant_resume = state['applicant_resume']


    result = extracted_personal_info_generator.invoke({"resume": applicant_resume})
    write_personal_info_file(result['name'], result['email'], result['contact_number'], result['linkedIn'])

    return {"applicant_resume":applicant_resume, "applicant_name":result['name'], "applicant_email":result['email']}



def change_pdf_name(state):

    print("---CHANGE PDF NAME---")
    applicant_resume_path = state['applicant_resume_path']


    try:
        base_path, file_name = os.path.split(applicant_resume_path)
        file_name, ext = os.path.splitext(file_name)
        new_file_name = f"{state['applicant_name']}{ext}"
        new_path = os.path.join(base_path, new_file_name)
        os.rename(applicant_resume_path, new_path)
    except Exception as e:
        print(f"Unexpected error In renaming: {e}")



def resume_score_calculator(state):

    print("---RESUME SCORE CALCULATE---")
    applicant_resume = state['applicant_resume']
    job_description = state['job_description']

    result = resume_score_generator.invoke({"resume": applicant_resume, "job":job_description})



    try:
        with open('scores.csv', 'a') as file:
            file.write(f"{state['applicant_name']},{state['applicant_email']},{result['resume_score']}\n")
    except IOError as e:
        print(f"Error writing to file: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")



workflow = StateGraph(GraphState)

workflow.add_node("extract_personal_info", extract_personal_info)
workflow.add_node("change_pdf_name", change_pdf_name)
workflow.add_node("resume_score_calculator", resume_score_calculator)



workflow.set_entry_point("extract_personal_info")


workflow.add_edge("extract_personal_info", "change_pdf_name")
workflow.add_edge("change_pdf_name", "resume_score_calculator")
workflow.add_edge("resume_score_calculator", END)


aapp = workflow.compile()


def log_improvements(email,name, improvement_msg, filename='improvement_msg.json'):
    try:
        data = {"email": email, "name":name,"improvement_msg": improvement_msg} 
        with open(filename, 'a') as json_file:
            json.dump(data, json_file)
            json_file.write('\n')

    except IOError as e:
        print(f"An I/O error occurred: {e}")

    except json.JSONDecodeError as e:
        print(f"A JSON decoding error occurred: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def log_selected(email, name, filename='selected.json'):
    try:
        data = {"email": email, "name": name} 
        with open(filename, 'a') as json_file:
            json.dump(data, json_file)
            json_file.write('\n')

    except IOError as e:
        print(f"An I/O error occurred: {e}")

    except json.JSONDecodeError as e:
        print(f"A JSON decoding error occurred: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")



resume_improvements_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an experienced Technical Human Resource Manager.
    your task is to review the provided resume against the job description. 

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Please provide a professional evaluation of how well the candidate's profile aligns with the role. 
    Highlight any areas where the applicant may need improvement in relation to the job requirements, and offer constructive suggestions to enhance their resume for this position.
    Identify the weaknesses of the applicant in relation to the specified job requirements. Give suggesions to improve there resume regarts to this job.
    Give as a paragraph at most 100 words.
    
    RESUME CONTENT:\n\n {resume} \n\n
    JOB DESCRIPTION:\n\n {job} \n\n
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["resume", "job"],
)

resume_improvments = resume_improvements_prompt | llm | StrOutputParser()



resume_summary_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an experienced Technical Human Resource Manager.
    your task is to review the provided resume ang give summary about person. 

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Please provide a summary of the candidate. Include suitable job area. Clearly identify candidate male or female.
    Give as a paragraph at most 100 words.
    
    RESUME CONTENT:\n\n {resume} \n\n
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["resume"],
)

resume_summary = resume_summary_prompt | llm | StrOutputParser()
##########################################################################################

def ai_image_generator(summary, name):
    
    def extract_keywords(text):
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        keywords = [word for word in words if word.isalnum() and word not in stop_words]
        freq_dist = FreqDist(keywords)
        return freq_dist.most_common(5)

    def generate_image(prompt):
        try:
            # Generate the image using the OpenAI API
            response = client.images.generate(model="dall-e-3",
            prompt=prompt,                                          
            #template=template_image_data,
            n=1,
            size="1024x1024")

            # Access the generated image URL
            image_url = response.data[0].url
            #print("Generated Image URL:", image_url)
        
            # Fetch the image from the URL
            image_response = requests.get(image_url)
            image = Image.open(BytesIO(image_response.content))
            return image
        
        except Exception as e:
            print(f"Error generating image: {e}")
            return None

        # List of 50 example pets
    pets = [
        "dog", "cat", "parrot", "rabbit", "hamster", "goldfish", "turtle", "snake",
        "lizard", "frog", "tarantula", "gerbil", "ferret", "guinea pig", "chinchilla",
        "mouse", "rat", "hedgehog", "canary", "lovebird", "cockatiel", "budgerigar",
        "macaw", "finch", "dove", "duck", "chicken", "goose", "turkey", "peacock",
        "goat", "sheep", "pig", "cow", "horse", "donkey", "llama", "alpaca", "pony",
        "crab", "octopus", "axolotl", "gecko", "iguana", "python", "boa constrictor",
        "scorpion", "beetle"
    ]
    
    # List of tech fields and associated descriptions
    fields = [
        ("AI", "perfect for an AI researcher exploring new algorithms"),
        ("Data Science", "ideal for a data scientist analyzing complex datasets"),
        ("Software Development", "a great companion for a software developer writing code"),
        ("Hardware Engineering", "suitable for a hardware engineer designing circuits"),
        ("Cybersecurity", "a good fit for a cybersecurity analyst securing systems"),
        ("Robotics", "an excellent match for a robotics engineer building robots"),
        ("Cloud Computing", "great for a cloud architect designing scalable solutions"),
        ("Machine Learning", "perfect for a machine learning engineer training models"),
        ("DevOps", "ideal for a DevOps engineer automating deployments"),
        ("Web Development", "a wonderful companion for a web developer crafting websites"),
        ("Mobile Development", "excellent for a mobile developer creating apps"),
        ("Game Development", "a good fit for a game developer designing games"),
        ("Networking", "ideal for a network engineer configuring systems"),
        ("Database Management", "perfect for a database administrator managing data"),
        ("Bioinformatics", "great for a bioinformatics expert analyzing genetic data"),
        ("Embedded Systems", "a great match for an embedded systems engineer coding firmware"),
        ("Blockchain", "suitable for a blockchain developer working on decentralized applications"),
        ("IoT", "perfect for an IoT developer connecting smart devices"),
        ("VR/AR", "ideal for a VR/AR developer creating immersive experiences"),
        ("Quantum Computing", "an excellent companion for a quantum computing researcher studying qubits")
    ]
    
    # Combine pets with fields and traits to create descriptions
    pet_descriptions = {}
    for i in range(1000):
        pet = random.choice(pets)
        field, description = random.choice(fields)
        trait1, trait2 = random.sample(fields, 2)
        tech_description = f"{trait1[1]} and {trait2[1].split(' ')[-1]}."
        full_description = f"{pet.capitalize()}: {description}. Additionally, it is {tech_description}"
        pet_descriptions[f"pet_{i+1}"] = full_description

    keywords = extract_keywords(summary)

    # Create a list of descriptions
    descriptions_list = list(pet_descriptions.values())
    
    # Include CV summary in the list for comparison
    texts = [summary] + descriptions_list
    
    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Compute cosine similarity between CV summary and all pet descriptions
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Find the index of the most similar description
    best_match_index = cosine_similarities.argmax()
    
    # Find the best matching description
    best_match_description = descriptions_list[best_match_index]

    # Define prompt and template image URL
    prompt = best_match_description
    
    # Generate the image
    output_image = generate_image(prompt)
    
    gen_img_directory = "genImg"

    # Create the directory if it doesn't exist
    os.makedirs(gen_img_directory, exist_ok=True)
    imgPath = os.path.join(gen_img_directory, f"{name}.png")
    # Save the image inside the genImg folder with the specified name
    output_image.save(imgPath)
    
    return output_image






#############################################################################################
@app.route('/process_data', methods=['GET'])
def process_data():
    job_loader = PyPDFLoader("./job.pdf")
    job_description = job_loader.load_and_split()


    folder_path = './data'

    pdf_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):

                full_path = os.path.join(root, file)
                pdf_files.append(full_path)

    for pdf in pdf_files:
        
        applicant_resume_path = pdf
        resume_loader = PyPDFLoader(applicant_resume_path)
        applicant_resume = resume_loader.load_and_split()

        inputs = {"applicant_resume_path": applicant_resume_path,"applicant_resume":applicant_resume,"job_description": job_description}
        output = aapp.invoke(inputs)


    

    return jsonify("yesssssss")

@app.route('/send_mail', methods=['GET'])
def send_mail():
    number_of_applicant_to_choose = int(request.args.get('ncand'))
    columns = ['Name', 'Email', 'Score']
    data = pd.read_csv("./scores.csv", header=None, names=columns)
    top_scores = data.nlargest(number_of_applicant_to_choose, 'Score')
    reject_resumes = data[~data.index.isin(top_scores.index)]

    job_loader = PyPDFLoader("./job.pdf")
    job = job_loader.load_and_split()

    for index, row in reject_resumes.iterrows():
        try:
            pdf_path = "./data/"+row["Name"]+".pdf"
            resume_loader = PyPDFLoader(pdf_path)
            resume = resume_loader.load_and_split()


            resume_improvments = resume_improvements_prompt | llm | StrOutputParser()

            result = resume_improvments.invoke({"resume": resume,"job":job})
            log_improvements(row["Email"], row["Name"],result)

            resume_summary = resume_summary_prompt | llm | StrOutputParser()
            summary = resume_summary.invoke({"resume": resume})
            ai_image_generator(summary, row["Name"])
                
        except Exception as e:
            print(f"Error opening or processing PDF '{row["Email"]}': {e}")
    
    for index, row in top_scores.iterrows():
        log_selected(row["Email"], row["Name"])
                
        



if __name__ == '__main__':
    app.run(debug=True)

