import streamlit as st
from page1 import medical_image_enhancement
from page2 import medical_image_analysis
from page3 import medical_symptom_checker
from page4 import lab_report_explainer

# Set page config
st.set_page_config(
    page_title="DocSight - AI Medical Assistant",
    page_icon="🧠",
    layout="wide"
)

# Custom header with emoji
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🧠 DocSight - Your AI Medical Assistant</h1>
    <p style='text-align: center; font-size: 18px; color: gray;'>Analyze, interpret, and understand your medical data using AI</p>
    <hr style="border: 1px solid #eee;" />
    """,
    unsafe_allow_html=True
)

# Sidebar with better visual style
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4320/4320337.png", width=120)
    st.title("🗂️ Navigation")
    st.markdown("---")
    page = st.radio(
        "Select a Feature:",
        [
            "🖼️ Medical Image Enhancement",
            "🔬 Medical Image Analysis",
            "🤒 Symptom Checker",
            "📄 Medical Interpreter"
        ]
    )
    st.markdown("---")
    st.info("Built with ❤️")

# Routing logic
if "Enhancement" in page:
    medical_image_enhancement()
elif "Analysis" in page:
    medical_image_analysis()
elif "Symptom" in page:
    medical_symptom_checker()
elif "Interpreter" in page:
    lab_report_explainer()




