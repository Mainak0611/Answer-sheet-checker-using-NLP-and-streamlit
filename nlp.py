import streamlit as st
import pdfplumber
import torch
import re
import matplotlib.pyplot as plt
import language_tool_python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util


# Initialize NLP tools
tool = language_tool_python.LanguageTool('en-US')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Streamlit UI
st.title(" Answer Sheet Checker System")
st.write("Upload both the answer key and student's answer sheet (PDF or TXT).")
col1, col2 = st.columns(2)
with col1:
    uploaded_key = st.file_uploader("Upload Answer Key (PDF/TXT)", type=["pdf", "txt"])
with col2:
    uploaded_student = st.file_uploader("Upload Student Answer Sheet (PDF/TXT)", type=["pdf", "txt"])

# Threshold Sliders
correct_threshold = st.slider("Set Correct Answer Threshold", 0.0, 1.0, 0.8, 0.05)
partial_threshold = st.slider("Set Partial Answer Threshold", 0.0, correct_threshold, 0.5, 0.05)


# Extract and Process Files
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")


# Enhanced Text Preprocessing
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if
              word not in stop_words]  # Lemmatization & Stopword removal
    return ' '.join(tokens)


def parse_qa(text):
    segments = re.split(r'Q\d+\.', text)
    qa_pairs = []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        parts = re.split(r'Answer:|A:', segment, flags=re.IGNORECASE)
        question_text = preprocess_text(parts[0].strip())
        answer_text = preprocess_text(parts[1].strip()) if len(parts) >= 2 else ""
        qa_pairs.append((question_text, answer_text))
    return qa_pairs


# Grading System
def grade_answers(key_qa, student_qa, correct_threshold=0.8, partial_threshold=0.5):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    results = []
    correct_count = 0
    partial_count = 0
    total = min(len(key_qa), len(student_qa))

    for i in range(total):
        key_question, key_answer = key_qa[i]
        stu_question, stu_answer = student_qa[i]

        emb_key = model.encode(key_answer, convert_to_tensor=True)
        emb_stu = model.encode(stu_answer, convert_to_tensor=True)

        cosine_score = util.pytorch_cos_sim(emb_key, emb_stu).item()

        if cosine_score >= correct_threshold:
            grade = " Correct"
            correct_count += 1
        elif cosine_score >= partial_threshold:
            grade = " Partially Correct"
            partial_count += 1
        else:
            grade = " Incorrect"

        # Grammar Check
        grammar_errors = len(tool.check(stu_answer))

        results.append({
            "question_number": i + 1,
            "key_answer": key_answer,
            "student_answer": stu_answer,
            "similarity": cosine_score,
            "grade": grade,
            "grammar_errors": grammar_errors
        })
    return correct_count, partial_count, total, results


# Visualization
def plot_results(correct, partial, incorrect):
    labels = ['Correct', 'Partially Correct', 'Incorrect']
    sizes = [correct, partial, incorrect]
    colors = ['#4CAF50', '#FFC107', '#F44336']

    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figure size
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%', colors=colors,
        startangle=90, shadow=True, wedgeprops={'edgecolor': 'black'}
    )

    # Improve text visibility
    for text in texts + autotexts:
        text.set_fontsize(14)
        text.set_color('black')

    ax.axis('equal')  # Ensures the pie chart is circular
    plt.tight_layout()  # Ensures proper alignment

    return fig


# Generate Automated Feedback
def generate_feedback(correct, partial, incorrect, total, results):
    feedback = "### AI-Generated Feedback\n\n"

    if correct == total:
        feedback += "Excellent work! You've answered all questions correctly. Keep it up!"
    elif correct > total * 0.7:
        feedback += f"Great job! You got {correct} out of {total} correct. Try reviewing the partially correct answers to improve further."
    elif correct > total * 0.5:
        feedback += f"Decent performance! You got {correct} out of {total} correct. Focus on improving the incorrect answers by revising key concepts."
    else:
        feedback += f"Needs Improvement! You got only {correct} out of {total} correct. Consider reviewing the study material and practicing more."

    avg_grammar_errors = sum(res['grammar_errors'] for res in results) / total if total else 0
    feedback += f"\n\n Grammar Check Summary: You had an average of {avg_grammar_errors:.1f} grammar errors per answer. Try to improve grammatical accuracy for better readability."

    return feedback


if st.button("Evaluate"):
    if uploaded_key and uploaded_student:
        with st.spinner("Processing... Please wait."):
            if uploaded_key.type == "application/pdf":
                key_text = extract_text_from_pdf(uploaded_key)
            else:
                key_text = extract_text_from_txt(uploaded_key)

            if uploaded_student.type == "application/pdf":
                student_text = extract_text_from_pdf(uploaded_student)
            else:
                student_text = extract_text_from_txt(uploaded_student)

            key_qa = parse_qa(key_text)
            student_qa = parse_qa(student_text)

            if not key_qa or not student_qa:
                st.error("Could not parse questions from the provided files.")
            else:
                correct_count, partial_count, total, results = grade_answers(key_qa, student_qa, correct_threshold,
                                                                             partial_threshold)
                incorrect_count = total - (correct_count + partial_count)

                st.success(f"Total questions graded: {total}")
                st.success(f" Correct answers: {correct_count}")
                st.warning(f" Partially Correct answers: {partial_count}")
                st.error(f" Incorrect answers: {incorrect_count}")
                st.success(f"Final Score: {correct_count} / {total}")

                # Results Table
                st.write("### Grading Breakdown")
                for res in results:
                    st.markdown(f"""
                        Question {res['question_number']}  
                        - Similarity Score: {res['similarity']:.2f} ({res['grade']})  
                        - Key Answer: {res['key_answer']}  
                        - Student Answer: {res['student_answer']}  
                        - Grammar Errors: {res['grammar_errors']} errors detected  
                        ---
                    """)

                # Show visualization
                fig = plot_results(correct_count, partial_count, incorrect_count)
                st.pyplot(fig)

                # AI-Generated Feedback
                st.markdown(generate_feedback(correct_count, partial_count, incorrect_count, total, results))