import os
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
import streamlit as st
from agno.tools.googlesearch import GoogleSearchTools
from agno.media import Image as AgnoImage


def lab_report_explainer():
    if "GOOGLE_API_KEY" not in st.session_state:
        st.session_state.GOOGLE_API_KEY = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "initial_summary" not in st.session_state:
        st.session_state.initial_summary = None

    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Configuration")

        if not st.session_state.GOOGLE_API_KEY:
            api_key = st.text_input("Enter your Google API Key:", type="password")
            if api_key:
                st.session_state.GOOGLE_API_KEY = api_key
                st.success("API Key saved!")
                st.rerun()
        else:
            st.success("API Key is configured")
            if st.button("🔄 Reset API Key"):
                st.session_state.GOOGLE_API_KEY = None
                st.rerun()

        st.selectbox(
            "🌐 Choose Output Language",
            ["English", "Hindi", "Kannada", "Spanish", "Tamil", "French"],
            key="selected_language",
            help="Select your preferred language for explanation"
        )

        st.warning(
            "⚠️ This tool is for educational purposes. Always consult a certified medical professional."
        )

    agent = Agent(
        model=Gemini(api_key=st.session_state.GOOGLE_API_KEY, id="gemini-2.0-flash"),
        tools=[GoogleSearchTools()],
        markdown=True
    ) if st.session_state.GOOGLE_API_KEY else None

    st.title("🧪 Lab Report and Doctor Prescription Interpreter")
    st.write("Upload a prescription or report and get a simplified, "
             "patient-friendly explanation in your preferred language.")

    uploaded_file = st.file_uploader(
        "📤 Upload Lab Report/ Prescription Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file and agent:
        image = PILImage.open(uploaded_file)
        st.image(image, caption="Uploaded Lab Report", use_container_width=True)

        if st.button("📊 Explain Report"):
            image_path = "temp_lab_report.png"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("🧠 Analyzing your lab report..."):
                try:
                    prompt = f"""
                    You are a compassionate medical assistant AI. Analyze the uploaded prescription or lab report image and provide a detailed explanation in {st.session_state.selected_language}.

                    Please include these 4 sections:

                    1. Prescription/Lab Report Summary  
                    - List all medications or tests, with dose, frequency, and duration.  
                    - Highlight critical or abnormal results.

                    2. Health Insights  
                    - Explain the purpose of each medication or test grouped by body system or function.  
                    - Describe urgency (Normal / Monitor / See doctor soon / Critical).

                    3. Patient-Friendly Explanation  
                    - Use simple, non-medical language and analogies.  
                    - Provide emotional reassurance to the patient.

                    4. Patient Resources  
                    - Provide 2-3 reliable, easy-to-understand online resources with URLs.  
                    - Suggest simple actions patients should take.

                    Guidelines:  
                    - Translate section headers and terms into {st.session_state.selected_language} when possible.  
                    - Avoid medical jargon or explain it simply.  
                    - Use Markdown formatting with clear headers, bullet points, and links.

                    Example output style:

                    ### 1. Prescription Summary  
                    - Augmentin 625mg Tablets: Take one tablet three times a day after meals for 5 days.  
                    - ...

                    ### 2. Health Insights  
                    - Infection: Augmentin is an antibiotic that fights bacteria.  
                    - Urgency: Monitor...

                    ### 3. Patient-Friendly Explanation  
                    Think of Augmentin as a "cleanup crew" that fights infection...

                    ### 4. Patient Resources  
                    - [Augmentin Info - Drugs.com](https://www.drugs.com/augmentin.html)  
                    - [Gum Health Tips - Healthline](https://www.healthline.com/health/dental-and-oral-health/ways-to-keep-gums-healthy)

                    Thank you.
                    """

                    agno_image = AgnoImage(filepath=image_path)
                    response = agent.run(prompt, images=[agno_image])

                    if st.session_state.initial_summary != response.content:
                        st.session_state.initial_summary = response.content
                        st.session_state.chat_history = []  # reset chat history for new report

                    st.subheader("🧾 Lab Report Summary")
                    st.markdown(st.session_state.initial_summary)

                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    os.remove(image_path)

    # 💬 Chat section
    if agent and st.session_state.initial_summary:
        st.markdown("---")
        st.subheader("💬 Ask Questions About Your Report")

        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)

        user_input = st.chat_input("Ask a follow-up question (e.g., 'Is my creatinine level serious?')")
        if user_input:
            with st.chat_message("🧑‍💻 You"):
                st.markdown(user_input)

            with st.spinner("🤖 Thinking..."):
                try:
                    # Combine previous report summary + new user question
                    contextual_prompt = f"""
                    You have the following lab report explanation:

                    {st.session_state.initial_summary}

                    User question: {user_input}
                    
                    You are a helpful, knowledgeable medical assistant who can:
                     
                    1. Answer detailed follow-up questions about lab tests, health conditions, and prescriptions.
                    2. When the user asks for external information like doctors, hospital recommendations, or resources not 
                    included in the lab report, you must use the Google Search tool to find relevant, up-to-date information.
                    """

                    chat_response = agent.run(contextual_prompt)
                    st.session_state.chat_history.append(("🧑‍💻 You", user_input))
                    if chat_response.content.strip():
                        st.session_state.chat_history.append(("🤖 AI", chat_response.content))
                        with st.chat_message("🤖 AI"):
                            st.markdown(chat_response.content)
                except Exception as e:
                    st.error(f"Error during chat: {e}")

        # Add the voice assistant call option
        st.markdown("---")
        st.subheader("Need further help?")
        if st.button("📞 Call a Doctor"):
            st.markdown("""
                You can speak to our doctors in a private room.

                🔗 [Click here to Talk to a Medical Expert](https://docsight-expert-connect.vercel.app/)
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    lab_report_explainer()
