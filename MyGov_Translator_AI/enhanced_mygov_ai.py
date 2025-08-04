import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI
import os
from deep_translator import GoogleTranslator
import re
import json
from typing import Dict, List, Optional
import logging

# 🆕 NEW: Added logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔧 IMPROVED: Better API key handling with caching and error checking
@st.cache_resource
def get_openai_client():
    """Initialize OpenAI client with caching"""
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OpenAI API key not found. Please add it to your secrets.")
        st.stop()
    return OpenAI(api_key=api_key)

# 🔧 IMPROVED: Better PDF text extraction with proper error handling
def extract_pdf_text(uploaded_file) -> str:
    """Extract text from PDF with improved error handling"""
    try:
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            page_text = page.get_text(sort=True)
            if page_text.strip():
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text
        
        pdf_document.close()
        
        if not text.strip():
            st.warning("⚠️ No text found in the PDF. This might be a scanned document that requires OCR.")
            return ""
            
        return text
    
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        st.error(f"❌ Error reading PDF: {str(e)}")
        return ""

# 🆕 NEW: Comprehensive scheme analysis function
def get_scheme_summary(client, text: str, max_chars: int = 10000) -> str:
    """Generate a comprehensive summary of the government scheme"""
    try:
        if len(text) > max_chars:
            text = text[:max_chars]
            last_period = text.rfind('.')
            if last_period > max_chars * 0.8:
                text = text[:last_period + 1]
        
        prompt = f"""
        You are an expert government policy analyst. Analyze this government scheme document and provide a comprehensive summary in the following format:

        **SCHEME NAME:** [Name of the scheme]

        **PURPOSE:** [What this scheme aims to achieve]

        **KEY BENEFITS:**
        • [Benefit 1]
        • [Benefit 2]
        • [Benefit 3]

        **ELIGIBILITY CRITERIA:**
        • [Criterion 1]
        • [Criterion 2]
        • [Criterion 3]

        **REQUIRED DOCUMENTS:**
        • [Document 1]
        • [Document 2]
        • [Document 3]

        **APPLICATION PROCESS:**
        1. [Step 1]
        2. [Step 2]
        3. [Step 3]

        **IMPORTANT DETAILS:**
        • Application deadline: [if mentioned]
        • Contact information: [if mentioned]
        • Subsidy/benefit amount: [if mentioned]

        Document content:
        {text}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return f"Error generating summary: {str(e)}"

# 🆕 NEW: Eligibility checking function
def check_eligibility(client, summary: str, user_profile: Dict) -> str:
    """Check user eligibility based on the scheme and user profile"""
    try:
        profile_text = f"""
        Age: {user_profile.get('age', 'Not specified')}
        Gender: {user_profile.get('gender', 'Not specified')}
        Income: {user_profile.get('income', 'Not specified')}
        Category: {user_profile.get('category', 'Not specified')}
        State: {user_profile.get('state', 'Not specified')}
        Occupation: {user_profile.get('occupation', 'Not specified')}
        Education: {user_profile.get('education', 'Not specified')}
        """
        
        prompt = f"""
        Based on the government scheme details and user profile below, determine eligibility and provide guidance.

        SCHEME DETAILS:
        {summary}

        USER PROFILE:
        {profile_text}

        Provide a response in this format:

        **ELIGIBILITY STATUS:** [ELIGIBLE/NOT ELIGIBLE/PARTIALLY ELIGIBLE]

        **EXPLANATION:**
        [Detailed explanation of why they are or aren't eligible]

        **IF NOT ELIGIBLE - STEPS TO BECOME ELIGIBLE:**
        1. [Step 1 if applicable]
        2. [Step 2 if applicable]
        3. [Step 3 if applicable]

        **NEXT STEPS:**
        [What the user should do next to apply or become eligible]

        **REQUIRED DOCUMENTATION:**
        [List documents they need to gather based on their profile]
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Error checking eligibility: {str(e)}")
        return f"Error checking eligibility: {str(e)}"

# 🔧 IMPROVED: Much better Hindi translation using GPT
def translate_to_hindi(client, text: str) -> str:
    """Translate to simple Hindi using GPT for better context understanding"""
    try:
        prompt = f"""
        Translate the following government scheme information to very simple Hindi that a common person, farmer, or villager can easily understand. 
        Use simple words and avoid complex technical terms. Make it conversational and easy to understand.
        
        Text to translate:
        {text}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Error translating to Hindi: {str(e)}")
        try:
            paragraphs = text.split('\n\n')
            translated_paragraphs = []
            
            for paragraph in paragraphs:
                if paragraph.strip():
                    translated = GoogleTranslator(source='auto', target='hi').translate(paragraph)
                    translated_paragraphs.append(translated)
            
            return '\n\n'.join(translated_paragraphs)
        except:
            return f"Translation error: {str(e)}"

# 🔧 IMPROVED: Better Telugu translation with sentence-by-sentence processing
def translate_to_telugu(text: str) -> str:
    """Translate to Telugu using improved chunking"""
    try:
        sentences = re.split(r'[.!?]+', text)
        translated_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 2:
                try:
                    translated = GoogleTranslator(source='auto', target='te').translate(sentence)
                    if translated:
                        translated_sentences.append(translated)
                except:
                    translated_sentences.append(sentence)
        
        return '. '.join(translated_sentences)
    
    except Exception as e:
        logger.error(f"Error translating to Telugu: {str(e)}")
        return f"Translation error: {str(e)}"

# 🆕 NEW: User profile collection function
def get_user_profile() -> Dict:
    """Collect user profile information for eligibility checking"""
    st.subheader("👤 Your Profile (for Eligibility Check)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=25)
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        income = st.selectbox("Annual Income", [
            "Below ₹1 Lakh", "₹1-3 Lakhs", "₹3-5 Lakhs", 
            "₹5-10 Lakhs", "₹10-20 Lakhs", "Above ₹20 Lakhs"
        ])
        category = st.selectbox("Category", ["General", "OBC", "SC", "ST", "EWS"])
    
    with col2:
        state = st.selectbox("State", [
            "Andhra Pradesh", "Telangana", "Tamil Nadu", "Karnataka", "Kerala",
            "Maharashtra", "Gujarat", "Rajasthan", "Uttar Pradesh", "Bihar",
            "West Bengal", "Odisha", "Madhya Pradesh", "Chhattisgarh", "Jharkhand",
            "Haryana", "Punjab", "Himachal Pradesh", "Uttarakhand", "Delhi",
            "Other"
        ])
        occupation = st.selectbox("Occupation", [
            "Farmer", "Student", "Government Employee", "Private Employee",
            "Self-employed", "Business Owner", "Unemployed", "Retired", "Other"
        ])
        education = st.selectbox("Education", [
            "Primary School", "High School", "Intermediate", "Graduate",
            "Post Graduate", "Professional Degree", "Others"
        ])
    
    return {
        "age": age,
        "gender": gender,
        "income": income,
        "category": category,
        "state": state,
        "occupation": occupation,
        "education": education
    }

# 🔧 IMPROVED: Better main function with enhanced UI
def main():
    st.set_page_config(
        page_title="MyGov Translator AI",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 MyGov Translator AI - Enhanced Version")
    st.markdown("*Upload government scheme documents and get simplified translations with eligibility checking*")
    
    # Initialize OpenAI client
    client = get_openai_client()
    
    # File upload
    uploaded_pdf = st.file_uploader(
        "Upload a Government Scheme PDF", 
        type=["pdf"],
        help="Upload a PDF document containing government scheme details"
    )
    
    if uploaded_pdf:
        st.info(f"📎 Uploaded: {uploaded_pdf.name} ({uploaded_pdf.size} bytes)")
        
        # Extract text
        with st.spinner("🔍 Extracting text from PDF..."):
            text = extract_pdf_text(uploaded_pdf)
        
        if text:
            with st.expander("📖 Extracted Text Preview", expanded=False):
                st.text_area("Preview", text[:2000] + "..." if len(text) > 2000 else text, height=200)
            
            # Get user profile for eligibility checking
            user_profile = get_user_profile()
            
            if st.button("🚀 Analyze Scheme & Check Eligibility", type="primary"):
                
                # Generate summary
                with st.spinner("📝 Analyzing scheme document..."):
                    summary = get_scheme_summary(client, text)
                
                # Display English summary
                st.subheader("📋 Scheme Summary (English)")
                st.markdown(summary)
                st.divider()
                
                # Check eligibility
                with st.spinner("✅ Checking your eligibility..."):
                    eligibility = check_eligibility(client, summary, user_profile)
                
                st.subheader("🎯 Your Eligibility Status")
                st.markdown(eligibility)
                st.divider()
                
                # Translations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🇮🇳 हिंदी अनुवाद")
                    with st.spinner("अनुवाद कर रहे हैं..."):
                        combined_text = f"{summary}\n\n--- आपकी पात्रता ---\n{eligibility}"
                        hindi_translation = translate_to_hindi(client, combined_text)
                    st.markdown(hindi_translation)
                
                with col2:
                    st.subheader("🇮🇳 తెలుగు అనువాదం")
                    with st.spinner("అనువదిస్తున్నాము..."):
                        combined_text = f"{summary}\n\n--- మీ అర్హత ---\n{eligibility}"
                        telugu_translation = translate_to_telugu(combined_text)
                    st.markdown(telugu_translation)
                
                st.success("✅ Analysis complete! Check the translations above for your language preference.")
                
        else:
            st.error("❌ Could not extract text from the PDF. Please ensure it's a text-based PDF, not a scanned image.")

if __name__ == "__main__":
    main()
