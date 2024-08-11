import os
from dotenv import load_dotenv
import streamlit as st
import PyPDF2
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize the Groq client
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_groq_response(input_text, pdf_content, prompt):
    try:
        combined_prompt = f"{input_text}\n\n{pdf_content}\n\n{prompt}\nPlease provide the response in plain text format."

        # Create the chat completion request
        chat_completion = groq.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": combined_prompt
            }
        ],
        model="llama3-8b-8192",
        temperature=0.6,  # Slightly lower for more factual responses
        # max_tokens=6000,  # Increase this to get longer and more detailed responses
        top_p=0.9,  # Adjust for more coherent responses
        frequency_penalty=0.2,  # Discourage repetition
        presence_penalty=0.4,  # Encourage introduction of new topics
        stream=False,
        response_format={"type": "text"}  # Request the response in plain text format
        )



        # Access the message content
        raw_text = chat_completion.choices[0].message.content
        return raw_text
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

def extract_text_from_pdf(uploaded_file):
    if uploaded_file is not None:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    else:
        raise FileNotFoundError("No file uploaded")

def handle_resume_evaluation(uploaded_file, input_text, prompt):
    if uploaded_file is not None:
        try:
            pdf_content = extract_text_from_pdf(uploaded_file)
            response = get_groq_response(input_text, pdf_content, prompt)
            st.subheader("The Response is:")
            st.write(response)
        except Exception as e:
            st.write(f"An error occurred: {str(e)}")
    else:
        st.write("Please upload the resume")
        
def handle_custom_prompt(custom_prompt):
    try:
        response = get_groq_response("", "", custom_prompt)
        st.subheader("Model Response:")
        st.write(response)
    except Exception as e:
        st.write(f"An error occurred: {str(e)}")

# Streamlit App
st.set_page_config(page_title=" ANURAGS LLAMA RESUME EXPERT")
st.header("ANURAGS LLAMA RESUME EXPERT")
input_text = st.text_area("Job description:", key="input")
uploaded_file = st.file_uploader("Upload your resume (PDF):", type=["pdf"])

if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")

input_prompt1 = """
You are an expert AI assistant specializing in analyzing resumes {uploaded_file}and extracting detailed information. I have uploaded a resume{uploaded_file}, and I need you to thoroughly analyze it. Your task is to:

**Provide a Summary**: Offer a brief overview of the candidate, including key skills, experience, and education.

**Project Details**: For each project mentioned in the resume, provide the following:
   - A detailed description of the project.
   - Identify any technologies, frameworks, or methodologies used.
   - If available, provide code snippets or pseudo-code relevant to the project.
   - Search for and suggest any potential GitHub repositories, YouTube videos, or other online resources that might be associated with the project (based on the project description and keywords).

**Additional Recommendations**: If possible, suggest how the candidate can improve the presentation of their projects or their resume overall, such as including links to relevant online resources or elaborating on certain aspects of the projects.

Please ensure that the analysis is thorough and that any provided links or code snippets are directly related to the projects mentioned in the resume{uploaded_file}.
""" 


input_prompt2 = """
You are an advanced Applicant Tracking System (ATS) with expertise in analyzing and comparing resumes{uploaded_file} against job descriptions{input_text}. I have provided a job description and a resume. Your task is to identify missing keywords that are critical for the job role but are not present in the resume.

1. **Analyze the Job Description**: Carefully review the job description and extract the essential keywords, skills, and qualifications required for the role.
   
2. **Compare with Resume**: Evaluate the provided resume and compare it with the job description to identify any missing keywords, skills, and experiences that are crucial for the role.

3. **List of Missing Keywords**: Provide a list of the missing keywords, skills, and qualifications that the resume should include to better match the job description.

4. **Suggestions**: Offer suggestions on how to naturally incorporate these missing keywords into the resume without compromising the authenticity of the candidate’s experience.

Make sure that your analysis is thorough, and that the missing keywords are relevant and critical to the job role.
"""


input_prompt3 = """
You are an expert in optimizing resumes to achieve the highest possible ATS (Applicant Tracking System) score. I have a resume{uploaded_file} and a job description{input_text} that I need you to analyze. Your task is to:

1. **Analyze the Job Description**: Identify the most important keywords, skills, and experiences that are essential for the job.
2. **Evaluate the Resume**: Review the projects, work experience, and skills listed in the resume to identify areas that can be enhanced to match the job description more closely.
3. **Add Keywords to Projects and Work Experience**: Tailor the resume by incorporating relevant keywords and phrases from the job description into the projects, work experience, and skills sections of the resume. This should be done naturally and in a way that accurately reflects the candidate's experience.
4. **Provide Specific Recommendations**: Suggest specific changes or additions to the resume, such as including particular technologies, methodologies, or achievements that align with the job description. Highlight areas where the resume can be expanded or refined to better match the job role.
5. **Final Optimized Resume**: Present a revised version of the resume that is optimized to achieve a higher ATS score, making the candidate more competitive for the job.

Ensure that the resume remains truthful and accurately represents the candidate's experience while maximizing alignment with the job description.

"""
input_prompt4 = """
You are a highly experienced Technical Interviewer and an expert in various technologies and software development practices. I have uploaded a resume{uploaded_file}, and I need you to generate a list of the top 100 interview questions along with detailed answers. The questions should be directly related to the candidate's projects, technologies mentioned, and their overall experience.

Specifically, focus on:
1. **Project-Based Questions**: Generate questions that dig deep into the specific projects listed on the resume. These should cover project objectives, challenges faced, solutions implemented, and outcomes.
   
2. **Technology-Focused Questions**: Provide questions related to the technologies, tools, and frameworks mentioned in the resume. Include questions that assess the candidate’s understanding, practical usage, and problem-solving capabilities with these technologies.

3. **Behavioral and Situational Questions**: Include questions that assess the candidate's ability to work in a team, handle project deadlines, manage conflicts, and adapt to new technologies or challenges.

The goal is to prepare the candidate for any interview related to their specific projects and technologies, ensuring they can confidently discuss their work and technical skills.

"""
input_prompt5 = """
You are a highly advanced job search assistant with deep knowledge of job matching algorithms and online job platforms such as LinkedIn, Google Jobs, Dice, and others. Based on the resume {uploaded_file} and job discription show me atleast 50 jobs I have uploaded, your task is to find new job openings that closely match the candidate's skills, experience, and career goals.
Also if some are mostly related if you find single skill matching also display it ***Display 50 jobs minimum***
For each job opening, provide the following:
1. **Job Title**: The title of the job that matches the candidate’s profile.
2. **Company Name**: The name of the company offering the job.
3. **Location**: The location of the job (remote, hybrid, or on-site).
4. **Job Description**: A brief description of the job, highlighting the key responsibilities and required skills.
5. **Application Link**: Provide direct links to the job listings on platforms such as LinkedIn, Google Jobs, Dice, and others where the candidate can apply.
6. **Company Website**: Include a link to the company's website or careers page for more information.

Ensure the job openings are recent and relevant, matching the candidate's experience in technologies like [Insert Key Technologies from Resume] and roles such as [Insert Relevant Job Titles from Resume].

"""

input_prompt6 = """
You are an advanced Applicant Tracking System (ATS) with a deep understanding of how to match resumes{uploaded_file} with job descriptions{input_text}.
Your task is to evaluate the provided resume against the following job description{input_text}. 
Calculate a match score as a percentage, indicating how well the resume aligns with the job requirements. 

Additionally, provide a detailed breakdown of:
1. **Matched Keywords**: Highlight the keywords or skills from the job description{input_text} that are present in the resume{uploaded_file} .
2. **Missing Keywords**: Identify important keywords or skills that are missing in the resume{uploaded_file}  but are critical for the job role.
3. **Final Thoughts**: Offer a brief summary of the overall alignment, including any suggestions for improving the resume {uploaded_file} to better match the job description{input_text}.


"""

# New Section: Custom Input Prompt
st.subheader("Custom Prompt Input")
custom_prompt = st.text_area("Enter your custom prompt here:", key="custom_prompt")

if st.button("Submit Custom Prompt"):
    handle_custom_prompt(custom_prompt)

if st.button("Tell me About my resume"):
    handle_resume_evaluation(uploaded_file, input_text, input_prompt1)

elif st.button("Identify missing Keywords in Resume"):
    handle_resume_evaluation(uploaded_file, input_text, input_prompt2)

elif st.button("Generate Tailored Resume Improvements"):
    handle_resume_evaluation(uploaded_file, input_text, input_prompt3)

elif st.button("Top 100 interview questions for my resume"):
    handle_resume_evaluation(uploaded_file, input_text, input_prompt4)

elif st.button("latest job openings Finder"):
    handle_resume_evaluation(uploaded_file, input_text, input_prompt5)

elif st.button("Percentage match"):
    handle_resume_evaluation(uploaded_file, input_text, input_prompt6)
