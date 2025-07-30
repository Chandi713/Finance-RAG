import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
import random
import html
from typing import Dict, Any, List
from RAG_Pipeline import qdrant_pipeline_instance, rag_pipeline_instance
from dotenv import load_dotenv

load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Fincumen",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Environment variables
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Custom CSS for styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #1f77b4;
    margin-bottom: 2rem;
}

.chat-container {
    max-height: 600px;
    overflow-y: auto;
    padding: 1rem;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    background-color: #f8f9fa;
}

.user-message-container {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    margin: 10px 0;
    margin-right: 10px;
}

.user-message {
    background-color: #007bff;
    color: white;
    padding: 10px 15px;
    border-radius: 15px 15px 5px 15px;
    text-align: left;
    max-width: 75%;
    min-width: 150px;
    width: fit-content;
    min-height: 20px;
    height: auto;
    word-wrap: break-word;
    word-break: break-word;
    overflow-wrap: break-word;
    white-space: pre-wrap;
    display: block;
    box-sizing: border-box;
    margin-bottom: 5px;
}

.assistant-message {
    background-color: #f1f3f4;
    color: #333;
    padding: 10px 15px;
    border-radius: 15px 15px 15px 5px;
    margin: 10px 0;
    margin-right: 20%;
}

.copy-button {
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 11px;
    color: #666;
    cursor: pointer;
    transition: all 0.2s ease;
    display: inline-flex;
    align-items: center;
    gap: 3px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    font-weight: 500;
    text-decoration: none;
    user-select: none;
    margin-top: 3px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

.copy-button:hover {
    background: #e9ecef;
    border-color: #adb5bd;
    color: #495057;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.copy-button:active {
    transform: translateY(0);
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

.copy-button.copied {
    background: #d4edda;
    border-color: #c3e6cb;
    color: #155724;
}

.copy-button-user {
    align-self: flex-end;
}

.copy-button-assistant {
    align-self: flex-start;
    margin-left: 0;
}

.assistant-response-container {
    background-color: #ffffff !important;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    line-height: 1.6;
    color: #333333 !important;
    width: 100%;
    max-width: 100%;
    height: auto;
    min-height: fit-content;
    overflow: visible;
    box-sizing: border-box;
}

.response-content {
    color: #333333 !important;
    font-size: 1rem;
    background-color: transparent !important;
}

.assistant-response-container * {
    color: #333333 !important;
}

.assistant-response-container h1,
.assistant-response-container h2,
.assistant-response-container h3,
.assistant-response-container h4,
.assistant-response-container h5,
.assistant-response-container h6 {
    color: #1f77b4;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
}

.assistant-response-container p {
    margin-bottom: 1em;
}

.assistant-response-container table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
}

.assistant-response-container th,
.assistant-response-container td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}

.assistant-response-container th {
    background-color: #f8f9fa;
    font-weight: bold;
}

.assistant-response-container ul,
.assistant-response-container ol {
    margin: 1em 0;
    padding-left: 2em;
    color: #333333 !important;
}

.assistant-response-container li {
    margin: 0.5em 0;
    color: #333333 !important;
    line-height: 1.5;
}

.assistant-response-container li strong {
    color: #1f77b4 !important;
    font-weight: 600;
}

.assistant-response-container blockquote {
    border-left: 4px solid #007bff;
    margin: 1em 0;
    padding: 0.5em 1em;
    background-color: #f8f9fa;
}

.assistant-response-container code {
    background-color: #f8f9fa;
    padding: 2px 4px;
    border-radius: 3px;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
}

.assistant-response-container pre {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 5px;
    padding: 1em;
    overflow-x: auto;
}

.assistant-response-container strong {
    font-weight: 600;
}

.assistant-response-container em {
    font-style: italic;
}

.assistant-response-container b,
.assistant-response-container strong {
    font-weight: bold !important;
    color: #1f77b4 !important;
}

.assistant-response-container i {
    font-style: italic !important;
    color: #666 !important;
}

.assistant-response-container br {
    line-height: 1.5;
}

.assistant-response-container [style] {
    color: inherit !important;
}

div[data-testid="stMarkdownContainer"] .assistant-response-container * {
    color: #333333 !important;
    background-color: transparent !important;
}

div[data-testid="stMarkdownContainer"] .assistant-response-container p {
    color: #333333 !important;
    margin-bottom: 1em !important;
}

div[data-testid="stMarkdownContainer"] .assistant-response-container b,
div[data-testid="stMarkdownContainer"] .assistant-response-container strong {
    color: #1f77b4 !important;
    font-weight: bold !important;
}

div[data-testid="stMarkdownContainer"] .assistant-response-container i {
    color: #666 !important;
    font-style: italic !important;
}

.assistant-response-container img {
    max-width: 100% !important;
    height: auto !important;
    border-radius: 5px !important;
    margin: 10px 0 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
}

.image-placeholder {
    background-color: #f8f9fa !important;
    border: 1px dashed #ccc !important;
    padding: 20px !important;
    text-align: center !important;
    border-radius: 5px !important;
    margin: 10px 0 !important;
    color: #666 !important;
    font-style: italic !important;
}

/* Sidebar styling for static, non-scrollable layout */
.css-1d391kg {
    overflow: hidden !important;
}

section[data-testid="stSidebar"] > div {
    overflow: hidden !important;
    padding-top: 1rem !important;
}

section[data-testid="stSidebar"] .block-container {
    overflow: hidden !important;
    max-height: 100vh !important;
}
</style>

<script>
function copyToClipboard(elementId, button) {
    const element = document.getElementById(elementId);
    const text = element.innerText || element.textContent;
    
    navigator.clipboard.writeText(text).then(function() {
        const originalText = button.innerHTML;
        button.innerHTML = '✓ Copied';
        button.classList.add('copied');
        
        setTimeout(function() {
            button.innerHTML = originalText;
            button.classList.remove('copied');
        }, 2000);
    }).catch(function(err) {
        console.error('Failed to copy text: ', err);
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        try {
            document.execCommand('copy');
            const originalText = button.innerHTML;
            button.innerHTML = '✓ Copied';
            button.classList.add('copied');
            
            setTimeout(function() {
                button.innerHTML = originalText;
                button.classList.remove('copied');
            }, 2000);
        } catch (err) {
            console.error('Fallback copy failed: ', err);
        }
        document.body.removeChild(textArea);
    });
}
</script>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables and auto-connect to RAG system."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "qdrant_connected" not in st.session_state:
        st.session_state.qdrant_connected = False
    if "current_metadata" not in st.session_state:
        st.session_state.current_metadata = {}
    if "initialization_status" not in st.session_state:
        st.session_state.initialization_status = {}
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "processing_query" not in st.session_state:
        st.session_state.processing_query = False
    
    auto_initialize_rag_system()

def check_environment_variables():
    """Check if required environment variables are set."""
    required_vars = ["QDRANT_URL", "QDRANT_API_KEY", "GEMINI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    return missing_vars

def test_qdrant_connection():
    """Test the Qdrant connection."""
    try:
        collections = qdrant_pipeline_instance.client.get_collections()
        return True, f"Connected successfully! Found {len(collections.collections)} collections."
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

def initialize_rag_system():
    """Initialize the RAG system."""
    try:
        if st.session_state.rag_system is None:
            st.session_state.rag_system = rag_pipeline_instance
        return True, "RAG system initialized successfully!"
    except Exception as e:
        return False, f"Failed to initialize RAG system: {str(e)}"

def get_company_name_from_ticker(ticker):
    """Get full company name from ticker symbol."""
    ticker_to_company = {
        "UNH": "UnitedHealth Group Incorporated", "PFE": "Pfizer Inc.",
        "JNJ": "Johnson & Johnson", "ABBV": "AbbVie Inc.",
        "LLY": "Eli Lilly and Company", "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation", "GOOGL": "Alphabet Inc.",
        "META": "Meta Platforms Inc.", "NVDA": "NVIDIA Corporation",
        "CAT": "Caterpillar Inc.", "GE": "General Electric Company",
        "MMM": "3M Company", "HON": "Honeywell International Inc.",
        "ITW": "Illinois Tool Works Inc.", "LMT": "Lockheed Martin Corporation",
        "NOC": "Northrop Grumman Corporation", "RTX": "Raytheon Technologies Corporation",
        "GD": "General Dynamics Corporation", "LHX": "L3Harris Technologies Inc.",
        "COF": "Capital One Financial Co.", "BAC": "Bank of America Corporation",
        "WFC": "Wells Fargo & Company", "MS": "Morgan Stanley"
    }
    
    if isinstance(ticker, list) and len(ticker) > 0:
        ticker = ticker[0]
    
    return ticker_to_company.get(ticker, ticker)

def auto_initialize_rag_system():
    """Automatically initialize RAG system on page load."""
    try:
        missing_vars = check_environment_variables()
        if missing_vars:
            st.session_state.initialization_status = {
                "env_check": False,
                "qdrant_connected": False,
                "rag_initialized": False,
                "error": f"Missing environment variables: {', '.join(missing_vars)}"
            }
            return
        
        qdrant_success, qdrant_message = test_qdrant_connection()
        st.session_state.qdrant_connected = qdrant_success
        
        rag_success, rag_message = False, "Qdrant connection failed"
        if qdrant_success:
            rag_success, rag_message = initialize_rag_system()
        
        st.session_state.initialization_status = {
            "env_check": True,
            "qdrant_connected": qdrant_success,
            "rag_initialized": rag_success,
            "qdrant_message": qdrant_message,
            "rag_message": rag_message,
            "timestamp": pd.Timestamp.now().strftime("%H:%M:%S")
        }
        
    except Exception as e:
        st.session_state.initialization_status = {
            "env_check": False,
            "qdrant_connected": False,
            "rag_initialized": False,
            "error": f"Initialization error: {str(e)}"
        }

def process_user_query(query: str) -> Dict[str, Any]:
    """Process user query through the RAG system."""
    try:
        if st.session_state.rag_system is None:
            return {"status": "error", "message": "RAG system not initialized"}
        
        result = st.session_state.rag_system.process_query(query)
        
        if result.get("current_metadata"):
            st.session_state.current_metadata = result["current_metadata"]
        
        return result
    except Exception as e:
        return {"status": "error", "message": f"Error processing query: {str(e)}"}

def process_images_in_content(content: str) -> str:
    """Process and secure image content in HTML."""
    import re
    
    if not content:
        return content
    
    img_pattern = r'<img[^>]*src=["\']([^"\']*)["\'][^>]*>'
    
    def replace_img_tag(match):
        full_tag = match.group(0)
        src_url = match.group(1)
        
        if src_url.startswith('data:image/'):
            return full_tag.replace('<img', '<img style="max-width: 100%; height: auto; border-radius: 5px;"')
        elif src_url.startswith(('http://', 'https://')):
            secure_tag = full_tag.replace('<img', '<img style="max-width: 100%; height: auto; border-radius: 5px;" loading="lazy"')
            return secure_tag
        elif src_url.startswith(('/', './')):
            return f'<div class="image-placeholder">Image: {src_url}</div>'
        else:
            return '<div class="image-placeholder">Image content</div>'
    
    content = re.sub(img_pattern, replace_img_tag, content, flags=re.IGNORECASE)
    return content

def detect_content_type(content: str) -> dict:
    """Detect what type of content we're dealing with."""
    import re
    
    content_info = {
        'has_images': bool(re.search(r'<img[^>]*src=', content, re.IGNORECASE)),
        'has_tables': bool(re.search(r'<table[^>]*>', content, re.IGNORECASE)),
        'has_charts': bool(re.search(r'<canvas[^>]*>|<script[^>]*chart', content, re.IGNORECASE)),
        'has_interactive': bool(re.search(r'<script[^>]*>', content, re.IGNORECASE)),
        'estimated_height': 300
    }
    
    content_length = len(content)
    line_count = content.count('\n') + content.count('<br>') + content.count('</p>')
    word_count = len(content.split())
    
    if content_info['has_charts'] or content_info['has_interactive']:
        base_height = 600
    elif content_info['has_tables']:
        table_rows = content.count('<tr>') 
        base_height = min(max(200 + (table_rows * 30), 250), 500)
    elif content_info['has_images']:
        base_height = 350
    else:
        estimated_text_height = max(
            150,
            min(
                80 + (line_count * 25) + (word_count * 2),
                600
            )
        )
        base_height = estimated_text_height
    
    content_info['estimated_height'] = max(150, min(base_height, 800))
    return content_info

def clean_html_content(content: str) -> str:
    """Clean and prepare HTML content for rendering."""
    if not content:
        return content
    
    import re
    
    # Security cleaning
    content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'javascript:', '', content, flags=re.IGNORECASE)
    content = re.sub(r'on\w+\s*=', '', content, flags=re.IGNORECASE)
    content = content.strip()
    
    # Remove markdown code block delimiters
    if content.startswith('```html') and content.endswith('```'):
        content = content[7:-3].strip()
    elif content.startswith('```html\n') and content.endswith('\n```'):
        content = content[8:-4].strip()
    elif content.startswith('```') and content.endswith('```'):
        first_newline = content.find('\n')
        if first_newline != -1 and first_newline < 10:
            content = content[first_newline+1:-3].strip()
        else:
            content = content[3:-3].strip()
    
    has_html_tags = bool(re.search(r'<[^>]+>', content))
    
    if has_html_tags:
        # Clean HTML structure
        content = re.sub(r'<!DOCTYPE[^>]*>', '', content, flags=re.IGNORECASE)
        content = re.sub(r'<html[^>]*>', '', content, flags=re.IGNORECASE)
        content = re.sub(r'</html>', '', content, flags=re.IGNORECASE)
        content = re.sub(r'<head[^>]*>.*?</head>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<body[^>]*>', '', content, flags=re.IGNORECASE)
        content = re.sub(r'</body>', '', content, flags=re.IGNORECASE)
        content = re.sub(r'\n\s*\n', '\n', content)
        content = content.strip()
        
        # Convert markdown formatting
        content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
        content = re.sub(r'\*(.*?)\*', r'<b>\1</b>', content)
        content = process_images_in_content(content)
        
        if not content or content.isspace():
            return '<div class="response-content"><p>No content available</p></div>'
        
        if not content.startswith('<'):
            content = f'<div class="response-content">{content}</div>'
        
        return content
    else:
        # Plain text conversion
        content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
        content = re.sub(r'\*(.*?)\*', r'<b>\1</b>', content)
        content = content.replace('\n\n', '</p><p>')
        content = content.replace('\n', '<br>')
        content = f'<div class="response-content"><p>{content}</p></div>'
        content = re.sub(r'<p></p>', '', content)
        content = re.sub(r'<p>\s*</p>', '', content)
        content = process_images_in_content(content)
    
    return content.strip()

def display_message(role: str, content: str, metadata: Dict = None, result_count: int = None, user_query: str = None):
    """Display a chat message with HTML content rendering."""
    if role == "user":
        # Enhanced responsive height calculation based on content length
        lines = content.count('\n') + 1
        words = len(content.split())
        chars = len(content)
        
        # More accurate height calculation considering text wrapping in fixed width
        # Assuming ~70 characters per line in the message bubble at max-width
        estimated_lines = max(lines, (chars // 70) + 1)
        estimated_words_lines = max(1, (words // 10))  # ~10 words per line estimate
        
        # Use the higher estimate for better accuracy
        effective_lines = max(estimated_lines, estimated_words_lines)
        
        # Base height: 25px per line + padding
        base_content_height = effective_lines * 25
        
        # Apply responsive thresholds with better scaling
        min_content_height = 48   # Minimum height matches CSS min-height
        max_content_height = 350  # Maximum to prevent excessive expansion
        
        # Calculate final content height
        responsive_content_height = max(min_content_height, min(base_content_height, max_content_height))
        
        # Add container padding and margins (32px padding + 40px margins)
        total_height = responsive_content_height + 72
        
        # Use components.html for user messages
        user_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                html, body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                    margin: 0;
                    padding: 10px;
                    background-color: transparent;
                    width: 100%;
                    box-sizing: border-box;
                    height: 100%;
                    overflow-x: clip;
                    overflow-y: visible;
                    clip-path: inset(0);
                }}
                .user-message-container {{
                    display: flex;
                    flex-direction: column;
                    align-items: flex-end;
                    justify-content: flex-start;
                    width: 100%;
                    margin: 20px 0;
                    margin-right: 15px;
                    position: relative;
                    min-height: fit-content;
                    height: auto;
                    box-sizing: border-box;
                    overflow: visible;
                }}
                .user-message {{
                    background-color: #007bff;
                    color: white;
                    padding: 16px;
                    border-radius: 15px 15px 15px 1px !important;
                    text-align: left;
                    max-width: 550px;
                    min-width: 180px;
                    width: fit-content;
                    min-height: 48px;
                    height: auto;
                    display: block;
                    box-sizing: border-box;
                    font-size: 1rem;
                    font-weight: 400;
                    position: relative;
                    box-shadow: 0 2px 8px rgba(0, 123, 255, 0.2);
                    transition: all 0.2s ease;
                    word-wrap: break-word;
                    word-break: break-word;
                    overflow-wrap: break-word;
                    hyphens: auto;
                    white-space: pre-wrap;
                    line-height: 1.6;
                }}
                .user-message:hover {{
                    box-shadow: 0 3px 12px rgba(0, 123, 255, 0.3);
                    transform: translateY(-1px);
                }}
                
                /* Ensure iframe contains all content properly */
                * {{
                    box-sizing: border-box;
                }}
            </style>
        </head>
        <body>
            <div class="user-message-container">
                <div class="user-message">{content}</div>
            </div>
        </body>
        </html>
        '''
        components.html(user_html, height=total_height, scrolling=False)
        
        # Spacing is now handled by container margin
        # st.markdown('<div style="margin-bottom: 15px;"></div>', unsafe_allow_html=True)
        
    else:
        html_content = clean_html_content(content)
        content_info = detect_content_type(html_content)
        
        # Create unique ID for the assistant message
        import hashlib
        message_id = hashlib.md5(content.encode()).hexdigest()[:8]
        
        with st.container():
            st.markdown('<div style="font-size: 1.5rem; font-weight: bold; color: #ffffff; margin: 1rem 0 0.5rem 0;">Assistant Response</div>', unsafe_allow_html=True)
            
            # Content type indicators
            indicators = []
            if content_info['has_images']:
                indicators.append("Images")
            if content_info['has_tables']:
                indicators.append("Tables")
            if content_info['has_charts']:
                indicators.append("Charts")
            if content_info['has_interactive']:
                indicators.append("Interactive")
            
            if indicators:
                st.caption(f"Content includes: {', '.join(indicators)}")
            
            # Combine content and button in single iframe with proper layout
            try:
                full_html = f'''
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        html, body {{ 
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                            line-height: 1.6;
                            color: #333;
                            background-color: transparent;
                            margin: 0;
                            padding: 0;
                            width: 100%;
                            height: auto;
                            box-sizing: border-box;
                            overflow: visible;
                        }}
                        .response-wrapper {{
                            position: relative;
                            width: 100%;
                            height: auto;
                            min-height: 100%;
                            margin-bottom: 0;
                        }}
                        .content-container {{
                            background-color: white;
                            border: 1px solid #e0e0e0;
                            border-radius: 10px;
                            padding: 20px 20px 15px 20px;
                            margin: 0;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            width: 100%;
                            box-sizing: border-box;
                            overflow-wrap: break-word;
                            word-wrap: break-word;
                        }}
                        .copy-button-external {{
                            background: rgba(248, 249, 250, 0.95);
                            border: 1px solid rgba(224, 224, 224, 0.8);
                            border-radius: 6px;
                            padding: 6px;
                            font-size: 14px;
                            color: #666;
                            cursor: pointer;
                            transition: all 0.2s ease;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            width: 28px;
                            height: 28px;
                            opacity: 0;
                            transform: translateY(5px);
                            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                            backdrop-filter: blur(10px);
                            user-select: none;
                            margin-top: 5px;
                            margin-left: 0;
                        }}
                        .response-wrapper:hover .copy-button-external {{
                            opacity: 1;
                            transform: translateY(0);
                        }}
                        .copy-button-external:hover {{
                            background: rgba(233, 236, 239, 0.95);
                            border-color: rgba(173, 181, 189, 0.9);
                            color: #495057;
                            transform: translateY(-1px) !important;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                        }}
                        .copy-button-external:active {{
                            transform: translateY(0) !important;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                        }}
                        .copy-button-external.copied {{
                            background: rgba(212, 237, 218, 0.95);
                            border-color: rgba(195, 230, 203, 0.9);
                            color: #155724;
                        }}
                        b, strong {{ font-weight: bold; color: #1f77b4; }}
                        i, em {{ font-style: italic; color: #666; }}
                        p {{ margin-bottom: 1em; line-height: 1.6; }}
                        br {{ line-height: 1.5; }}
                        img {{ max-width: 100%; height: auto; border-radius: 5px; margin: 10px 0; }}
                        table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f8f9fa; font-weight: bold; }}
                        .image-placeholder {{ 
                            background-color: #f8f9fa; 
                            border: 1px dashed #ccc; 
                            padding: 20px; 
                            text-align: center; 
                            border-radius: 5px; 
                            margin: 10px 0; 
                        }}
                    </style>
                    <script>
                        function copyToClipboard(elementId, button) {{
                            const element = document.getElementById(elementId);
                            const assistantText = element.innerText || element.textContent;
                            
                            // Get user query from data attribute
                            const userQuery = button.getAttribute('data-user-query');
                            
                            // Combine user query and assistant response
                            let combinedText = '';
                            if (userQuery && userQuery.trim() !== '') {{
                                combinedText = 'Query: ' + userQuery + '\\n\\n' + 'Response: ' + assistantText;
                            }} else {{
                                combinedText = assistantText;
                            }}
                            
                            navigator.clipboard.writeText(combinedText).then(function() {{
                                const originalHTML = button.innerHTML;
                                button.innerHTML = '✓';
                                button.classList.add('copied');
                                
                                setTimeout(function() {{
                                    button.innerHTML = originalHTML;
                                    button.classList.remove('copied');
                                }}, 2000);
                            }}).catch(function(err) {{
                                console.error('Failed to copy text: ', err);
                                // Fallback for older browsers
                                const textArea = document.createElement('textarea');
                                textArea.value = combinedText;
                                document.body.appendChild(textArea);
                                textArea.select();
                                try {{
                                    document.execCommand('copy');
                                    const originalHTML = button.innerHTML;
                                    button.innerHTML = '✓';
                                    button.classList.add('copied');
                                    
                                    setTimeout(function() {{
                                        button.innerHTML = originalHTML;
                                        button.classList.remove('copied');
                                    }}, 2000);
                                }} catch (err) {{
                                    console.error('Fallback copy failed: ', err);
                                }}
                                document.body.removeChild(textArea);
                            }});
                        }}
                    </script>
                </head>
                <body>
                    <div class="response-wrapper">
                        <div class="content-container" id="assistant-msg-{message_id}">
                            {html_content}
                        </div>
                        <button class="copy-button-external" onclick="copyToClipboard('assistant-msg-{message_id}', this)" data-user-query="{html.escape(user_query or '', quote=True)}">
                            📋
                        </button>
                    </div>
                </body>
                </html>
                '''
                components.html(full_html, height=content_info['estimated_height'] + 20, scrolling=True)
                
            except Exception as e:
                st.error(f"HTML rendering failed: {e}")
                st.text(content)

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">Fincumen</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #a6a6a6; font-size: 1.2rem; margin-top: -2rem; margin-bottom: 2rem; font-style: italic;">"Fusing Finance, Context, and Precision via Advanced RAG Systems"</p>', 
                unsafe_allow_html=True)
    
    # Sidebar with two distinct sections
    with st.sidebar:
        # Main Section (Primary Area) - FAQ Feature
        # Custom CSS for FAQ styling only
        st.markdown("""
        <style>
        /* FAQ specific styling */
        section[data-testid="stSidebar"] div[data-testid="stExpander"] {
            border: 1px solid #4a5568 !important;
            border-radius: 8px !important;
            margin-bottom: 8px !important;
            background-color: #2d3748 !important;
        }
        
        section[data-testid="stSidebar"] div[data-testid="stExpander"] > div[role="button"] {
            background-color: #2d3748 !important;
            color: white !important;
            padding: 12px !important;
            border-radius: 8px !important;
        }
        
        section[data-testid="stSidebar"] div[data-testid="stExpander"] > div[role="button"]:hover {
            background-color: #4a5568 !important;
        }
        
        section[data-testid="stSidebar"] div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
            background-color: #1a202c !important;
            color: #e2e8f0 !important;
            padding: 12px !important;
            border-radius: 0 0 8px 8px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # FAQ Container with proper height and scrolling
        with st.container(height=450, border=True):
            # FAQ Section Header
            st.markdown("### ❓ Frequently Asked Questions")
            
            # FAQ Items with collapsible expanders
            with st.expander("🚀 How do I get started?"):
                st.write("""
                **Getting started is simple:**
                
                1. **Ask a Question**: Type your financial query in the chat box below
                2. **Use Examples**: Click any example question to see how it works
                3. **Review Results**: The system will search through financial documents and provide detailed answers
                4. **Check Context**: Monitor the sidebar to see which companies and years are being referenced
                """)
            
            with st.expander("🏢 Which companies are supported?"):
                st.write("""
                **Our database includes major companies such as:**
                
                • **Technology**: Apple (AAPL), Microsoft (MSFT), NVIDIA (NVDA), Meta (META), Alphabet (GOOGL)\n
                • **Healthcare**: Johnson & Johnson (JNJ), Pfizer (PFE), UnitedHealth (UNH), AbbVie (ABBV), Eli Lilly (LLY)\n
                • **Financial**: Bank of America (BAC), Wells Fargo (WFC), Capital One (COF), Morgan Stanley (MS)\n
                • **Industrial**: Caterpillar (CAT), General Electric (GE), 3M (MMM), Honeywell (HON)\n
                • **Defense**: Lockheed Martin (LMT), Northrop Grumman (NOC), Raytheon (RTX)\n
                
                *And many more Fortune 500 companies...*
                """)
            
            with st.expander("📊 What types of questions can I ask?"):
                st.write("""
                **You can ask about various financial topics:**

                • **Financial Performance**: Revenue, earnings, profit margins, growth rates\n
                • **Financial Position**: Assets, liabilities, equity, debt ratios\n
                • **Cash Flow**: Operating, investing, financing activities\n
                • **Comparative Analysis**: Compare metrics across companies or time periods\n
                • **Trend Analysis**: Multi-year performance and growth patterns\n
                • **Risk Factors**: Business risks, regulatory challenges \n
                • **Strategic Insights**: Business segments, market expansion, R&D investments \n
                """)
            
            with st.expander("🎯 How accurate is the information?"):
                st.write("""
                **Our data accuracy is high because:**
                
                • **Source**: Information comes directly from official SEC filings (10-K, 10-Q reports) \n
                • **Recent**: Documents are regularly updated with latest filings \n
                • **Verified**: All data points are extracted from audited financial statements \n
                • **Context-Aware**: The system provides specific document references \n
                • **Cross-Referenced**: Multiple sources validate the same information \n
                
                *Always verify critical decisions with original SEC documents.*
                """)
            
            with st.expander("⏰ What time periods are covered?"):
                st.write("""
                **Time coverage includes:**
                
                • **Recent Years**: 2020-2024 (most comprehensive) \n
                • **Historical Data**: Some companies have data going back further \n
                • **Quarterly Reports**: Q1, Q2, Q3, Q4 filings when available \n
                • **Annual Reports**: Complete fiscal year data \n
                • **Real-time Updates**: New filings added as they become available \n
                
                *The specific years available vary by company and filing type.*
                """)
        
        # Bottom Section (Compact Area) - Components positioned at bottom left
        # Add CSS to position content at bottom left without changing section styling
        st.markdown("""
        <style>
        /* Target the 120px height container in sidebar and make it blend seamlessly */
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
            border: none !important;
            background: transparent !important;
            padding: 0 !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] div[style*="height: 120px"] {
            border: none !important;
            background: transparent !important;
            padding: 0 !important;
            box-shadow: none !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-end !important;
            align-items: flex-start !important;
        }
        
        /* Position context content at bottom left */
        .context-content {
            width: 100%;
            text-align: left;
            margin-bottom: 5px;
        }
        
        /* Position bottom section components wrapper */
        .bottom-section-wrapper {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            justify-content: flex-end;
            width: 100%;
            min-height: 0;
        }
        
        /* Style for Clear Context button positioning */
        .bottom-button {
            margin-top: 8px;
            align-self: flex-start;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Bottom section wrapper for bottom-left positioning
        st.markdown('<div class="bottom-section-wrapper">', unsafe_allow_html=True)
        
        # Context content container positioned at bottom left
        with st.container(height=120):
            # Create a spacer to push content to bottom
            st.markdown('<div style="flex-grow: 1;"></div>', unsafe_allow_html=True)
            
            # Context content wrapper
            with st.container():
                st.markdown('<div class="context-content">', unsafe_allow_html=True)
                
                metadata = st.session_state.get('current_metadata', {})
                company_metadata = metadata.get('Company', None)
                years_metadata = metadata.get('Year', None)
                
                company_display = None
                years_display = None
                
                if company_metadata and isinstance(company_metadata, list) and len(company_metadata) > 0:
                    # Convert each ticker to company name and display all
                    company_names = [get_company_name_from_ticker(ticker) for ticker in company_metadata]
                    company_display = ", ".join(company_names)
                
                if years_metadata and isinstance(years_metadata, list) and len(years_metadata) > 0:
                    # Display the actual year metadata fetched
                    sorted_years = sorted(years_metadata)
                    years_display = ", ".join(map(str, sorted_years))
                
                # Display context information
                if company_display or years_display:
                    if company_display:
                        st.write(f"**Company:** {company_display}")
                    
                    if years_display:
                        st.write(f"**Years:** {years_display}")
                else:
                    st.info("No context data available")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Clear Context button with bottom positioning class
        st.markdown('<div class="bottom-button">', unsafe_allow_html=True)
        if st.button("Clear Context"):
            if st.session_state.rag_system:
                st.session_state.rag_system.clear_context()
                st.session_state.current_metadata = {}
                st.success("Context cleared!")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close bottom-section-wrapper
    
    # Main chat interface
    if not st.session_state.rag_system:
        if "initialization_status" in st.session_state and st.session_state.initialization_status:
            status = st.session_state.initialization_status
            if "error" in status:
                st.error("System Initialization Failed")
                st.error(status["error"])
                st.info("Please refresh the page to retry initialization.")
            elif not status.get("rag_initialized", False):
                st.warning("System is initializing automatically...")
                st.info("Please wait for the RAG system to connect and initialize.")
        else:
            st.info("Initializing Fincumen...")
            st.info("The system is automatically connecting to Qdrant and initializing.")
        
        return
    
    # Chat interface
    st.subheader("Chat with Financial Documents")
    
    # Show interactive example queries on initial page load (when no messages exist)
    if not st.session_state.messages:
        st.markdown("Click on any example below to get started:")
        
        # Comprehensive set of example queries
        all_queries = [
            "What was Microsoft's total Revenue for the fiscal year 2022? Also, check for Apple too.",
            "Analyze the trend in Total assets and Total liabilities from fiscal year 2020 to 2022. Based on this, infer if Microsoft's financial leverage has increased or decreased over these periods, and what this might suggest about the company's financial risk.",
            "Compare Apple's effective tax rate to the statutory federal tax rate for 2024. Explain the key reasons behind any difference. Additionally, calculate the percentage increase in Research and Development (R&D) expenses from 2023 to 2024 and discuss the primary drivers for this increase as reported by the company.",
            "Between fiscal years 2020 and 2022, how did Microsoft's stock-based compensation expense and related income tax benefits evolve, and what was the average tax benefit rate as a percentage of expense across these three years? Additionally, discuss what this trend might suggest about Microsoft's equity compensation practices and their implications for financial planning.",
            "How do regulatory restrictions in China and India, combined with risks related to intellectual property protection and fulfillment network challenges, potentially limit Amazon's international growth and operational efficiency?",
            "What types of products are included in Johnson & Johnson's Consumer Health segment?",
            "What was Wells Fargo's earnings per common share (basic and diluted) for the year ended December 31, 2021?",
            "What was Microsoft revenue for the year ended December 31, 2021?",
            "Compare Johnson & Johnson's net cash flows from investing activities in 2020 and 2021. What were the major factors contributing to the shift in investing cash flow between these two years?",
            "Considering the company's steady net earnings, significant stock repurchases and dividends, and reduced net cash flow from investing activities in 2021, evaluate whether Johnson & Johnson prioritized shareholder returns over capital investments in that year. Support your answer with relevant financial data and trends visible across the cash flow statements from 2019–2021",
            "Which GE Aerospace board committee is responsible for overseeing cybersecurity risk, and who are the key management individuals that report to this committee on cybersecurity matters?",
            "Given the mention of \"heightened threats in connection with the separation of HealthCare and Vernova\" and the overall discussion of strategic risks, how might its Aerospace's cybersecurity risk management and strategy need to adapt to mitigate potential financial or operational impacts specifically related to data integrity and supply chain vulnerabilities arising from these recent spin-offs?",
            "Considering NVIDIA's \$34.0 billion share repurchase of 310 million shares in FY 2025, the additional \$50 billion buyback authorization in August 2024, and the significant increase in stock price from \$978.42 to \$2,287.06 over the five-year period ending January 26, 2025, analyze how the company's capital return strategy, through repurchases and dividends, may have contributed to enhancing shareholder value. Integrate relevant figures from stock performance, dividend payouts, and share withholding to support your conclusion.",
            "Considering NVIDIA's significant revenue growth in Data Center, driven by AI and accelerated computing, alongside the increasing complexity and frequency of new product introductions (e.g., Blackwell architecture, annual computing solutions cadence), what potential long-term financial implications might arise if the US government's expanding export controls, particularly the \"AI Diffusion\" IFR and potential restrictions on China-developed technologies, lead to a sustained and material inability for NVIDIA to sell its cutting-edge products in key international markets, and how might this impact their ability to manage inventory provisions and maintain their gross margin given the high R&D investments and long manufacturing lead times?",
            "Given Capital One Financial Corporation's financial data, calculate the change in the ratio of 'Total Cash and Cash Equivalents' to 'Total Liabilities' from December 31, 2022, to December 31, 2023. Additionally, determine the proportion of 'Net cash from operating activities' relative to 'Total liabilities' for both the fiscal year ended December 31, 2022, and December 31, 2023. Based on these calculations, what insights can be drawn regarding the company's short-term liquidity and its ability to cover its obligations from its core operations in the most recent fiscal year compared to the prior year?"
        ]
        
        # Randomly select 5 queries for this session
        if "selected_examples" not in st.session_state:
            st.session_state.selected_examples = random.sample(all_queries, 5)
        
        examples = st.session_state.selected_examples
        
        # Create clickable buttons for each example
        # Disable example buttons during query processing
        is_processing = (st.session_state.get('pending_query') is not None or 
                        st.session_state.get('processing_query', False))
        
        cols = st.columns(1)
        with cols[0]:
            for i, example in enumerate(examples):
                if st.button(example, key=f"example_{i}", use_container_width=True, disabled=is_processing):
                    # Add user message
                    st.session_state.messages.append({
                        "role": "user",
                        "content": example
                    })
                    # Set as pending query for processing
                    st.session_state.pending_query = example
                    st.rerun()
        
        st.markdown("---")
    
    # User input - Always display first to ensure visibility
    # Disable input during query processing to enforce single-query-at-a-time interaction
    is_processing = (st.session_state.get('pending_query') is not None or 
                    st.session_state.get('processing_query', False))
    
    user_input = st.chat_input(
        "Ask about financial data..." if not is_processing else "Processing your query...",
        disabled=is_processing
    )
    
    # Handle user input first
    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        st.session_state.pending_query = user_input
        st.rerun()
    
    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        user_query = None
        # If this is an assistant message, find the previous user message
        if message["role"] == "assistant" and i > 0:
            # Look for the most recent user message before this assistant message
            for j in range(i-1, -1, -1):
                if st.session_state.messages[j]["role"] == "user":
                    user_query = st.session_state.messages[j]["content"]
                    break
        
        display_message(
            role=message["role"],
            content=message["content"],
            metadata=message.get("metadata"),
            result_count=message.get("result_count"),
            user_query=user_query
        )
        
        # Add separator after assistant messages
        if message["role"] == "assistant":
            st.markdown("---")
    
    # Process pending query with spinner shown below messages
    if st.session_state.pending_query and not st.session_state.processing_query:
        st.session_state.processing_query = True
        query_to_process = st.session_state.pending_query
        st.session_state.pending_query = None
        
        # Show spinner below the user message during processing
        with st.spinner("Searching financial documents..."):
            result = process_user_query(query_to_process)
        
        # Handle response types
        if result["status"] == "success":
            assistant_message = {
                "role": "assistant",
                "content": result["content_summary"],
                "metadata": result.get("current_metadata", {}),
                "result_count": result.get("num_results", 0)
            }
        elif result["status"] == "prompt_needed":
            assistant_message = {
                "role": "assistant",
                "content": result["message"]
            }
        elif result["status"] == "error":
            assistant_message = {
                "role": "assistant",
                "content": f"Error: {result['message']}"
            }
        
        st.session_state.messages.append(assistant_message)
        st.session_state.processing_query = False
        st.rerun()

if __name__ == "__main__":
    main() 
