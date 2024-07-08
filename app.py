import streamlit as st
import requests
import pandas as pd
import os

sucess = False
# Set page configuration
st.set_page_config(
    page_title="Happy Hire Streamlit App",
    page_icon=":sparkles:",
    layout="wide"
)

# Sidebar for navigation
with st.sidebar:
    st.image("./assets/happyHire.jpg", use_column_width=True)
    



col1, col2 = st.columns(2)

with col1:
    uploaded_files1 = st.file_uploader("Choose Resumes To Analyze", accept_multiple_files=True)

with col2:
    uploaded_files2 = st.file_uploader("Choose Job Description File", accept_multiple_files=False)

    # Add a button below the file pickers, centered between them
col_spacer1, col_center, col_spacer2 = st.columns([3, 1, 3])
with col_center:
    analyze_button = st.button("   Analyze Resumes   ")

if analyze_button:
    if uploaded_files1 and uploaded_files2:
        response = requests.get('http://localhost:5000/process_data')
        if response.status_code == 200:
            pass

        else:
            st.error('Failed to process data.')
    else:
        st.warning("Please upload files in both fields to proceed.")

if os.path.exists("scores.csv"):
    columns = ['Name', 'Email', 'Score']
    df = pd.read_csv("./scores.csv", header=None, names=columns)
    df = df.sort_values(by='Score', ascending=False)
    df = df.reset_index(drop=True)
    st.subheader("Processed Scores")
    st.table(df)    
    
    num_candidates = st.number_input("Select number of top candidates to choose", min_value=1,
                                                    max_value=len(df), value=1, step=1)

    if st.button("Send Emails to Candidates"):
        num_candidates = str(num_candidates)
        response = requests.get('http://localhost:5000/send_mail' + "?ncand="+num_candidates )
        if response.status_code == 200:
            sucess = True
        

st.success("Mails send")