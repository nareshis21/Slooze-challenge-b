import streamlit as st
import os
import traceback
import pandas as pd
import json
import time

from dotenv import load_dotenv
from agent.llm_client import GroqClient
from agent.agent import LlamaPDFAgent as PDFAgent, AgentRateLimitError

# Load environment variables

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Naresh AI DocuPulse Submission - PDF Intelligence",
    page_icon="📄",
    layout="wide",
)

# Custom Styling for a Premium Dark Mode (Consistent with Challenge A)
st.markdown("""
<style>
    /* Main container styling - Deep Dark Gradient */
    .stApp {
        background: radial-gradient(circle at top left, #1e293b 0%, #0f172a 100%) !important;
        color: #f1f5f9 !important;
    }
    
    /* Header and Title styling - Neon Blue */
    h1 {
        color: #60a5fa !important;
        font-family: 'Outfit', sans-serif;
        font-weight: 800 !important;
        letter-spacing: -0.05rem;
        text-shadow: 0 0 20px rgba(96, 165, 250, 0.3);
    }
    
    h3 {
        color: #94a3b8 !important;
        font-weight: 400 !important;
    }

    /* Input styling - Darker Glass */
    .stTextInput>div>div>input {
        background-color: rgba(30, 41, 59, 0.7) !important;
        color: white !important;
        border: 1px solid rgba(96, 165, 250, 0.5) !important;
        border-radius: 12px !important;
        padding: 12px 20px !important;
        font-size: 1.1rem !important;
    }
    
    /* Button styling - Glowing Blue */
    .stButton>button {
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 15px 30px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 15px rgba(37, 99, 235, 0.4) !important;
        width: 100% !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 0 30px rgba(59, 130, 246, 0.6) !important;
    }

    /* Result Card styling - Dark Inset */
    .answer-container {
        background-color: rgba(30, 41, 59, 0.5);
        padding: 30px;
        border-radius: 20px;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.2);
        border-left: 8px solid #2563eb;
        margin-top: 25px;
    }
    
    /* Sidebar Dark Glass */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    .brand-text {
        font-size: 1.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #60a5fa, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    
    /* Standard Text Color Fixes */
    .stMarkdown, p, li {
        color: #cbd5e1 !important;
    }
    
    strong {
        color: #f1f5f9 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "pdf_agent" not in st.session_state:
    st.session_state.pdf_agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "deep_insights" not in st.session_state:
    st.session_state.deep_insights = {}


# Sidebar
with st.sidebar:
    st.markdown('<div class="brand-text">NARESH AI</div>', unsafe_allow_html=True)
    st.title("Settings")
    
    # API Key Input
    groq_api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    
    # Dynamic Model Fetching
    available_models = ["meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
    if groq_api_key:
        try:
            temp_client = GroqClient(api_key=groq_api_key)
            fetched_models = temp_client.list_models()
            if fetched_models:
                available_models = fetched_models
        except Exception:
            pass
            
    model_choice = st.selectbox(
        "Model Architecture",
        available_models,
        index=0 if "meta-llama/llama-4-scout-17b-16e-instruct" not in available_models else available_models.index("meta-llama/llama-4-scout-17b-16e-instruct")
    )

    
    st.divider()
    st.markdown("### 🗂️ Document Library")
    
    # Initialize agent if not exist (for library access)
    if "pdf_agent" in st.session_state and st.session_state.pdf_agent:
        if not hasattr(st.session_state.pdf_agent, "get_library"):
            st.session_state.pdf_agent = None # Clear stale object
            
    if not st.session_state.pdf_agent:
        from agent.agent import LlamaPDFAgent as PDFAgent
        st.session_state.pdf_agent = PDFAgent(api_key=groq_api_key or os.getenv("GROQ_API_KEY"), model=model_choice)
    
    library = st.session_state.pdf_agent.get_library()
    if not library:
        st.caption("No documents in library.")
    else:
        for doc in library:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.markdown(f"**{doc['filename']}**")
            with col2:
                if st.button("🗑️", key=f"del_{doc['hash']}", help="Delete vectors"):
                    if st.session_state.pdf_agent.delete_document(doc['hash']):
                        st.session_state.pdf_agent = None # Force re-init if active one deleted
                        st.rerun()
        st.info("To switch document, simply upload it again. It will load instantly from the library.")

    st.divider()
    st.markdown("### Document Controls")
    if st.button("Reset Session"):
        st.session_state.pdf_agent = None
        st.session_state.messages = []
        st.session_state.deep_insights = {}
        st.rerun()



    st.divider()
    st.markdown("### Profile")
    st.write("**Built by:** Naresh Kumar Lahajal")
    st.write("**Role:** GenAI Enthusiast")
    st.info("High-speed PDF intelligence powered by Groq and FastEmbed.")

# Header
st.title("Naresh AI DocuPulse - Submission")
st.subheader("Challenge B: PDF RAG & Summarization")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file and (st.session_state.pdf_agent is None or uploaded_file.name != st.session_state.get("last_uploaded_file")):
    with st.status("Ingesting document and indexing knowledge...", expanded=True) as status:
        try:
            agent = PDFAgent(api_key=groq_api_key, model=model_choice)
            status_msg = agent.ingest_pdf(uploaded_file)
            st.session_state.pdf_agent = agent
            st.session_state.last_uploaded_file = uploaded_file.name
            # Sync tables for explorer
            st.session_state.extracted_tables = agent.tables
            # Auto-Clear History on New Upload
            st.session_state.messages = []
            st.session_state.deep_insights = {}
            status.update(label=f"✅ {status_msg}", state="complete", expanded=False)
            st.toast("Intelligence Engine Initialized", icon="🧠")

        except Exception as e:
            st.error(f"Critical Ingestion Error: {e}")
            with st.expander("Show Traceback"):
                st.code(traceback.format_exc())


# Helper for Exact Backoff
def run_with_exact_backoff(func, *args, **kwargs):
    """
    Runs a function and catches AgentRateLimitError to perform a precise UI countdown retry.
    """
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except AgentRateLimitError as e:
            if attempt == max_attempts - 1:
                st.error(f"Failed after {max_attempts} attempts due to Persistent Rate Limits. Please wait a few minutes.")
                raise e
            
            # Precise wait + 1s buffer
            wait_time = int(e.wait_time) + 1
            st.toast(f"Rate Limit Hit! Waiting {wait_time}s to retry...", icon="⏳")
            
            # Visual Countdown
            placeholder = st.empty()
            for remaining in range(wait_time, 0, -1):
                placeholder.warning(f"⚠️ API Cooldown: Retrying in {remaining} seconds...")
                time.sleep(1)
            placeholder.empty()
    return None

if st.session_state.pdf_agent:

    # Action Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Ask Questions", "📝 Auto-Summary", "🧠 Deep Intelligence", "📋 Table Explorer"])

    
    with tab1:
        st.markdown("### 💬 Document Conversation")
        st.caption("Ask questions about the document and maintain a conversation thread.")
        
        # Display Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("🔗 Sources & Citations", expanded=False):
                        for i, src in enumerate(message["sources"]):
                            page_text = f"Page {src['page']}" if src['page'] else "Unknown Page"
                            st.markdown(f"**[{i+1}] {page_text}**")
                            st.caption(f"_{src['text']}_")
                            st.divider()

        # Chat Input
        if prompt := st.chat_input("What would you like to know?"):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing document context..."):
                    response_data = run_with_exact_backoff(st.session_state.pdf_agent.answer_question, prompt)
                    if response_data:
                        # Use st.write_stream for typing effect
                        answer = st.write_stream(response_data['answer_gen'])
                        sources = response_data.get("sources", [])
                        
                        if sources:
                            with st.expander("🔗 Sources & Citations", expanded=False):
                                for i, src in enumerate(sources):
                                    page_text = f"Page {src['page']}" if src['page'] else "Unknown Page"
                                    st.markdown(f"**[{i+1}] {page_text}**")
                                    st.caption(f"_{src['text']}_")
                                    st.divider()
                        
                        # Add assistant response to history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources
                        })




                
    with tab2:
        if st.button("Generate Executive Summary"):
            with st.spinner("Synthesizing document overview..."):
                streaming_response = run_with_exact_backoff(st.session_state.pdf_agent.summarize_document)
                if streaming_response:
                    st.markdown('<div class="answer-container" style="border-left: 8px solid #60a5fa;">', unsafe_allow_html=True)
                    st.markdown("### 📝 Document Summary")
                    st.write_stream(streaming_response.response_gen)
                    st.markdown('</div>', unsafe_allow_html=True)



    with tab3:
        st.markdown("### 🚀 Strategic Deep Analysis")
        st.info("This mode uses multi-stage recursive retrieval to extract deep strategic insights and KPIs.")
        
        if st.button("Run Deep Intelligence Scan"):
            with st.status("Analyzing document layers...", expanded=True) as status:
                st.write("🔍 Extracting Strategic Vision...")
                insights = run_with_exact_backoff(st.session_state.pdf_agent.get_deep_insights)
                if insights:
                    st.session_state.deep_insights = insights
                    
                    # Fetch KPI visualization data
                    st.write("📊 Generating Visual Analytics...")
                    viz_data = run_with_exact_backoff(st.session_state.pdf_agent.get_kpi_viz_data)
                    st.session_state.kpi_viz_data = viz_data
                    
                    status.update(label="✅ Deep Analysis Complete", state="complete", expanded=False)
                else:
                    status.update(label="❌ Failed after retries", state="error", expanded=False)


            
        if st.session_state.deep_insights:
            insights = st.session_state.deep_insights
            
            # 1. Strategic Vision
            st.markdown('<div class="answer-container" style="border-left: 8px solid #8b5cf6;">', unsafe_allow_html=True)
            st.markdown("#### 🎯 Strategic Vision")
            st.write(insights.get("strategic_vision", "N/A"))
            st.markdown('</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 2. Key Metrics
                st.markdown("#### 📊 Key Performance Indicators")
                metrics_text = insights.get("key_metrics", "")
                st.markdown(metrics_text if metrics_text else "No metrics extracted.")
                
            with col2:
                # 3. Risks
                st.markdown("#### ⚠️ Risks & Challenges")
                risks_text = insights.get("risks_and_challenges", "")
                st.markdown(risks_text if risks_text else "No risks identified.")
            
            # Visual Dashboard Section
            if st.session_state.get("kpi_viz_data"):
                st.divider()
                st.markdown("#### 📈 Key Trends & Metrics")
                viz_df = pd.DataFrame(st.session_state.kpi_viz_data)
                
                # Heuristic for chart type
                if any("year" in str(l).lower() or "q1" in str(l).lower() or "q2" in str(l).lower() or "q3" in str(l).lower() or "q4" in str(l).lower() for l in viz_df['label']):
                    st.line_chart(viz_df.set_index('label'), color="#3b82f6")
                    st.caption("Auto-detected Time Series data.")
                else:
                    st.bar_chart(viz_df.set_index('label'), color="#60a5fa")
                    st.caption("Bar chart representation of extracted KPIs.")

            # 4. SWOT Analysis

            st.divider()
            st.markdown("#### 🛠️ Automated SWOT Analysis")
            swot_raw = insights.get("swot_analysis", "{}")
            try:
                # Attempt to clean potential markdown artifacts around JSON
                if "```json" in swot_raw:
                    swot_raw = swot_raw.split("```json")[1].split("```")[0].strip()
                elif "{" in swot_raw:
                    swot_raw = "{" + swot_raw.split("{", 1)[1].rsplit("}", 1)[0] + "}"
                
                swot_data = json.loads(swot_raw)
                
                # Display SWOT in a grid
                s_col1, s_col2 = st.columns(2)
                with s_col1:
                    st.success(f"**Strengths**\n\n{swot_data.get('S', 'N/A')}")
                    st.info(f"**Opportunities**\n\n{swot_data.get('O', 'N/A')}")
                with s_col2:
                    st.warning(f"**Weaknesses**\n\n{swot_data.get('W', 'N/A')}")
                    st.error(f"**Threats**\n\n{swot_data.get('T', 'N/A')}")
            except Exception as e:
                st.write("Raw SWOT Insight:")
                st.write(swot_raw)
            
            # Report Export
            st.divider()
            report_md = f"""# Executive Intelligence Report: {st.session_state.last_uploaded_file}
            
## 🎯 Strategic Vision
{insights.get('strategic_vision', 'N/A')}

## 📊 Key Performance Indicators
{insights.get('key_metrics', 'N/A')}

## ⚠️ Risks & Challenges
{insights.get('risks_and_challenges', 'N/A')}

## 🛠️ SWOT Analysis
### Strengths
{swot_data.get('S', 'N/A') if 'swot_data' in locals() else 'N/A'}

### Weaknesses
{swot_data.get('W', 'N/A') if 'swot_data' in locals() else 'N/A'}

### Opportunities
{swot_data.get('O', 'N/A') if 'swot_data' in locals() else 'N/A'}

### Threats
{swot_data.get('T', 'N/A') if 'swot_data' in locals() else 'N/A'}

---
*Report generated by Naresh AI DocuPulse*
"""
            st.download_button(
                label="📥 Download Executive Intelligence Report",
                data=report_md,
                file_name=f"Intelligence_Report_{st.session_state.last_uploaded_file.replace('.pdf', '')}.md",
                mime="text/markdown"
            )

    with tab4:
        st.markdown("### 📋 PDF Table Explorer")
        st.info("Direct extraction of tabular data from the document. Select a table to explore.")
        
        tables = st.session_state.pdf_agent.tables
        if not tables:
            st.warning("No structured tables were detected in the document.")
        else:
            table_labels = [f"{t['label']} (Page Grounded)" for t in tables]
            selected_label = st.selectbox("Select Table", table_labels)
            
            # Find the selected table
            selected_idx = table_labels.index(selected_label)
            selected_table = tables[selected_idx]
            
            st.markdown(f"#### {selected_table['label']}")
            st.dataframe(selected_table['df'], width="stretch")
            
            # Download as CSV
            csv = selected_table['df'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download {selected_table['label']} as CSV",
                data=csv,
                file_name=f"{selected_table['label'].replace(' ', '_')}.csv",
                mime="text/csv"
            )




else:
    st.info("Please upload a PDF document to begin analysis.")


# Footer
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #64748b; padding: 20px;">
        © 2026 <b>Naresh Kumar Lahajal</b>. All Rights Reserved.<br>
        <small>Powered by Groq and Retrieval-Augmented Generation</small>
    </div>
    """,
    unsafe_allow_html=True
)
