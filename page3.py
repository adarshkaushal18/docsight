import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini

# Must be the first Streamlit command
st.set_page_config(
    page_title="AI Symptom Checker with Risk Assessment",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded"
)

def medical_symptom_checker():
    if 'assessment_result' not in st.session_state:
        st.session_state.assessment_result = {}
    if 'assessment_done' not in st.session_state:
        st.session_state.assessment_done = False
    if 'qa_pairs' not in st.session_state:
        st.session_state.qa_pairs = []

    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
        }
        .success-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f0fff4;
            border: 1px solid #9ae6b4;
        }
        .warning-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #fffaf0;
            border: 1px solid #fbd38d;
        }
        div[data-testid="stExpander"] div[role="button"] p {
            font-size: 1.1rem;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("💉 AI Symptom Checker with Risk Assessment")
    st.markdown("""
        <div style='background-color: #ff6347; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
        Get an AI-powered analysis of your symptoms and receive potential conditions with risk assessments.
        </div>
    """, unsafe_allow_html=True)

    # Sidebar for API Key
    with st.sidebar:
        st.header("🔑 API Configuration")
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Gemini API key to access the service"
        )

        if not gemini_api_key:
            st.warning("⚠️ Please enter your Gemini API Key to proceed")
            st.markdown("[Get your API key here](https://aistudio.google.com/apikey)")
            return

        st.success("API Key accepted!")

    # Main function logic
    if gemini_api_key:
        try:
            gemini_model = Gemini(id="gemini-1.5-flash", api_key=gemini_api_key)
        except Exception as e:
            st.error(f"❌ Error initializing Gemini model: {e}")
            return

        st.header("👤 Your Symptom Profile")

        col1, col2 = st.columns(2)

        with col1:
            symptoms_input = st.text_area("Symptoms (comma-separated)", help="Enter your symptoms")
            age = st.number_input("Age", min_value=1, max_value=120, step=1, help="Enter your age")
            gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
            lifestyle = st.text_input("Lifestyle Factors (e.g., smoking, alcohol consumption)",
                                      help="Enter relevant lifestyle factors")

        with col2:
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, step=0.1)
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, step=0.1)
            medical_history = st.text_area("Medical History", help="Any previous conditions or treatments")

        if st.button("🎯 Check Symptoms", use_container_width=True):
            with st.spinner("Analyzing your symptoms..."):
                try:
                    # Create an agent for symptom checking
                    symptom_checker_agent = Agent(
                        name="Symptom Checker",
                        role="Analyzes user symptoms and assesses risk",
                        model=gemini_model,
                        instructions=[
                            "Analyze the user's symptoms and suggest potential medical conditions.",
                            "Provide a risk level (e.g., low, moderate, high) based on the symptoms provided.",
                            "Consider age, gender, lifestyle factors, and medical history in the assessment.",
                            "Be clear, concise, and informative."
                        ]
                    )

                    user_profile = f"""
                    Symptoms: {symptoms_input}
                    Age: {age}
                    Weight: {weight}kg
                    Height: {height}cm
                    Gender: {gender}
                    Lifestyle: {lifestyle}
                    Medical History: {medical_history}
                    """

                    # Get symptom assessment from the agent
                    symptom_assessment_response = symptom_checker_agent.run(user_profile)
                    assessment_result = {
                        "conditions": symptom_assessment_response.content,
                        "risk_level": "Moderate (Based on age, symptoms, and medical history)",
                        "important_considerations": """
                        - Consult a healthcare provider for further evaluation.
                        - Pay attention to any worsening symptoms or new developments.
                        - Stay hydrated and maintain a balanced diet.
                        """
                    }

                    st.session_state.assessment_result = assessment_result
                    st.session_state.assessment_done = True
                    st.session_state.qa_pairs = []

                    display_symptom_assessment(assessment_result)

                except Exception as e:
                    st.error(f"❌ An error occurred while analyzing symptoms: {e}")

        if st.session_state.assessment_done:
            st.header("❓ Have Questions about Your Assessment?")
            question_input = st.text_input("What would you like to know?")

            if st.button("Get Answer"):
                if question_input:
                    with st.spinner("Finding the best answer for you..."):
                        assessment_result = st.session_state.assessment_result

                        context = f"Conditions: {assessment_result.get('conditions', '')}\nRisk Level: {assessment_result.get('risk_level', '')}"
                        full_context = f"{context}\nUser Question: {question_input}"

                        try:
                            # Create an agent for answering questions
                            agent = Agent(model=gemini_model, show_tool_calls=True, markdown=True)
                            run_response = agent.run(full_context)

                            if hasattr(run_response, 'content'):
                                answer = run_response.content
                            else:
                                answer = "Sorry, I couldn't generate a response at this time."

                            st.session_state.qa_pairs.append((question_input, answer))
                        except Exception as e:
                            st.error(f"❌ An error occurred while getting the answer: {e}")

            if st.session_state.qa_pairs:
                st.header("💬 Q&A History")
                for question, answer in st.session_state.qa_pairs:
                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {answer}")


# Function to display symptom assessment
def display_symptom_assessment(assessment_content):
    with st.expander("🔍 Symptom Assessment Results", expanded=True):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 🩺 Potential Conditions")
            st.write(assessment_content.get("conditions", "No conditions found"))

        with col2:
            st.markdown("### ⚠️ Risk Level & Recommendations")
            st.write(assessment_content.get("risk_level", "No risk level provided"))
            st.markdown("### 📝 Important Considerations")
            considerations = assessment_content.get("important_considerations", "").split('\n')
            for consideration in considerations:
                if consideration.strip():
                    st.warning(consideration)





