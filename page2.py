import os
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
import streamlit as st
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage


def medical_image_analysis():
    if "GOOGLE_API_KEY" not in st.session_state:
        st.session_state.GOOGLE_API_KEY = None

    with st.sidebar:
        st.title("ℹ️ Configuration")

        if not st.session_state.GOOGLE_API_KEY:
            api_key = st.text_input(
                "Enter your Google API Key:",
                type="password"
            )
            if api_key:
                st.session_state.GOOGLE_API_KEY = api_key
                st.success("API Key saved!")
                st.rerun()
        else:
            st.success("API Key is configured")
            if st.button("🔄 Reset API Key"):
                st.session_state.GOOGLE_API_KEY = None
                st.rerun()

        st.info("""
            This tool provides AI-Powered analysis of medical imaging data using
            advanced computer vision and radiological expertise.
            """
        )

        st.warning(
            "⚠️ Disclaimer: This tool is not a substitute for professional medical advice, diagnosis, or treatment."
            "Always seek the advice of qualified health professionals."
            "Do not make medical decisions based solely on the output of this tool."

        )

    medical_agent = Agent(
        model=Gemini(
            api_key=st.session_state.GOOGLE_API_KEY,
            id="gemini-2.0-flash"
        ),
        tools=[DuckDuckGoTools()],
        markdown=True
    ) if st.session_state.GOOGLE_API_KEY else None

    if not medical_agent:
        st.warning("Please configure your API key in the sidebar to continue")

    st.title("🏥 Medical Imaging Diagnosis Agent")
    st.write("Upload a medical image for professional analysis")

    uploaded_file = st.file_uploader(
        "Upload Medical Image",
        type=["jpg", "jpeg", "png", "dicom"],
        help="Supported formats: JPG, JPEG, PNG, DICOM"
    )

    if uploaded_file and medical_agent:
        image = PILImage.open(uploaded_file)
        st.image(image, caption="Uploaded Medical Image", use_container_width=True)

        if st.button("🔍 Analyze Image"):
            image_path = "temp_medical_image.png"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("🔄 Analyzing image... Please wait."):
                try:
                    query = """
                    You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the patient's medical image and structure your response as follows:

                    ### 1. Image Type & Region
                    - Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
                    - Identify the patient's anatomical region and positioning
                    - Comment on image quality and technical adequacy
                    
                    ### 2. Key Findings
                    - List primary observations systematically
                    - Note any abnormalities in the patient's imaging with precise descriptions
                    - Include measurements and densities where relevant
                    - Describe location, size, shape, and characteristics
                    - Rate severity: Normal/Mild/Moderate/Severe
                    
                    ### 3. Diagnostic Assessment
                    - Provide primary diagnosis with confidence level
                    - List differential diagnoses in order of likelihood
                    - Support each diagnosis with observed evidence from the patient's imaging
                    - Note any critical or urgent findings
                    
                    ### 4. Patient-Friendly Explanation
                    - Explain the findings in simple, clear language that the patient can understand
                    - Avoid medical jargon or provide clear definitions
                    - Include visual analogies if helpful
                    - Address common patient concerns related to these findings
                    
                    ### 5. Research Context
                    IMPORTANT: Use the DuckDuckGo search tool to:
                    - Find recent medical literature about similar cases
                    - Search for standard treatment protocols
                    - Provide a list of relevant medical links of them too
                    - Research any relevant technological advances
                    - Include 2-3 key references to support your analysis
                    
                    Format your response using clear markdown headers and bullet points. Be concise yet thorough.
                    """

                    agno_image = AgnoImage(filepath=image_path)
                    response = medical_agent.run(query, images=[agno_image])
                    st.markdown(response.content)
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    os.remove(image_path)
