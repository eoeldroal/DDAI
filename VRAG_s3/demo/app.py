# import streamlit as st
# from vrag_agent import VRAG
# from PIL import Image
# from time import sleep

# # Set page configuration
# st.set_page_config(
#     page_title="VRAG: Discovering More in Depth",
#     page_icon="🔍",  # Use a magnifying glass as the page icon
#     layout="wide",   # Wide layout
#     initial_sidebar_state="expanded"
# )

# def typewriter_effect(container, text, delay=0.02):
#     # This function simulates a typewriter effect by gradually appending characters
#     final_text = ''
#     for char in text:
#         # Append character
#         final_text += char
#         # Safeguard HTML brackets
#         text_for_display = final_text.replace("<", "&lt;").replace(">", "&gt;")
#         # Output to streamlit
#         container.markdown(f'<div class="info-box">{text_for_display}</div>', unsafe_allow_html=True)
#         sleep(delay)
#     container.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)
    


# def main():
#     # Page title
#     st.title("🔍 VRAG: Discovering More in Depth")
#     st.markdown("""
#     <style>
#         .main-title {
#             font-size: 36px;
#             color: #2E86C1;
#             text-align: center;
#             margin-bottom: 20px;
#         }
#         .header-text {
#             font-size: 20px;
#             color: #34495E;
#             margin-bottom: 10px;
#         }
#         .sidebar-header {
#             font-size: 18px;
#             color: #1ABC9C;
#             margin-bottom: 15px;
#         }
#         .submit-button {
#             background-color: #2E86C1;
#             color: white;
#             border-radius: 5px;
#             padding: 10px 20px;
#             font-size: 16px;
#         }
#         .divider {
#             border-top: 2px solid #1ABC9C;
#             margin-top: 5px;
#             margin-bottom: 5px;
#         }
#         .success-message {
#             background-color: #D5F5E3;
#             color: #1E8449;
#             padding: 10px;
#             border-radius: 5px;
#             margin-top: 10px;
#         }
#         .info-box {
#             font-size: 18px;  /* Increase the font size */
#             background-color: #E3F2FD; /* Light blue background */
#             padding: 15px;
#             border-left: 6px solid #2E86C1;
#             margin-bottom: 10px;
#             color: #1E2A38;
#         }
#         .small-image img {
#             display: block;
#             margin-left: auto;
#             margin-right: auto;
#             max-width: auto;
#             max-height: 500px;
#             text-align: center;
#         }
#         .caption {
#             font-size: 16px;
#             color: black; /* Black color for caption */
#             text-align: left; /* Change text alignment to left */
#             font-weight: bold; /* Bold caption */
#             margin-left: 10px;
#             margin-top: 10px;
#             margin-bottom: -5px; /* Passive bottom margin */
#         }
#     </style>
#     """, unsafe_allow_html=True)

#     # Initialize VRAG agent
#     agent = VRAG(
#     planner_model_path="/root/workspace/VRAG_test/VRAG_lsm/grpo_model/30_step_checkpoint"
#     )

#     # Sidebar configuration
#     with st.sidebar:
#         st.markdown('<p class="sidebar-header">⚙️ Configuration Options</p>', unsafe_allow_html=True)
#         MAX_ROUNDS = st.number_input('Number of Max Reasoning Iterations:', min_value=3, max_value=10, value=10, step=1)
#         selected_example = st.selectbox(
#             'Examples:', 
#             ["What is the most commonly used travel app?", 
#             "What is the usage rate of travel agents for leisure in the country with the highest credit card penetration rate in Southeast Asia?",
#             "助力证券公司上云的核心技术如何保证安全？", 
#             "互联网IPV6活跃用户占比在2019年是多少，到2023年增加了多少？"])
#         st.markdown('<hr style="border:1px solid #1ABC9C">', unsafe_allow_html=True)  # Divider line

#     # Question input box and button placed at the top of the page
#     st.markdown('<p class="header-text">📝 Enter Your Question:</p>', unsafe_allow_html=True)

#     agent.max_steps = MAX_ROUNDS
#     question = st.text_input(
#         "Question Input:",
#         placeholder="Type your question here...",
#         key="question_input",
#         value=selected_example,  # Set the default question here
#         label_visibility="collapsed"
#     )
#     col1, col2 = st.columns([1, 5])  # Adjust button layout
#     with col1:
#         submit_button = st.button("Submit", key="submit_button", help="Click to submit the question")
#     with col2:
#         if submit_button and question:
#             st.markdown('<p class="success-message">Question submitted! Processing...</p>', unsafe_allow_html=True)

#     # Create a scrollable container to display results
#     result_container = st.container()
#     with result_container:
#         if submit_button and question:
#             generator = agent.run(question)
#             image_width = 350
#             try:
#                 while True:
#                     action, content, raw_content = next(generator)
#                     if action == 'think':
#                         sleep(0.5)
#                         think = f"💭 Thinking: {content}"
#                     elif action == 'search':
#                         typewriter_effect(st.empty(), f'{think} <br> 🔍 <strong>Call Search Engine: {content}</strong>')
                        
#                         # st.markdown(f'<div class="info-box">{think} <br> 🔍 Searching: {content}</div>', unsafe_allow_html=True)
#                     elif action == 'bbox':
#                         bbox_str = content
#                         typewriter_effect(st.empty(), f'{think} <br> 📷 <strong>Region of Interest: {content}</strong>')
#                         # st.markdown(f'<div class="info-box">{think} <br> 📷 Region of Interest: {content}</div>', unsafe_allow_html=True)
#                     elif action == 'search_image':
#                         col1, col2 = st.columns(2)
#                         with col1:
#                             st.markdown(f'<p class="caption">Retrieved Image</p>', unsafe_allow_html=True)
#                             st.markdown('<div class="small-image">', unsafe_allow_html=True)
#                             st.image(content, width=image_width)
#                             st.markdown('</div>', unsafe_allow_html=True)
#                     elif action == 'crop_image':
#                         col1, col2 = st.columns(2)
#                         with col1:
#                             st.markdown(f'<p class="caption">🤔 Image with bbox: <span style="color:purple;">{bbox_str}</span></p>', unsafe_allow_html=True)
#                             st.markdown('<div class="small-image">', unsafe_allow_html=True)
#                             st.image(raw_content, width=image_width) 
#                             st.markdown('</div>', unsafe_allow_html=True)
#                         sleep(0.5)
#                         with col2:
#                             st.markdown(f'<p class="caption">✂️ Cropped Region</p>', unsafe_allow_html=True)
#                             st.markdown('<div class="small-image">', unsafe_allow_html=True)
#                             st.image(content, width=image_width) 
#                             st.markdown('</div>', unsafe_allow_html=True)
#             except StopIteration as e:
#                 action, content, raw_response = e.value
#                 if action == 'answer':
#                     st.success(f"✅ Answer: {content}")

# if __name__ == "__main__":
#     main()

import streamlit as st
from vrag_agent import VRAG
from PIL import Image
from time import sleep
import uuid # ⬅️ [변경] 기본 ID 생성을 위해 추가

# Set page configuration
st.set_page_config(
    page_title="VRAG: Discovering More in Depth",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def typewriter_effect(container, text, delay=0.02):
    final_text = ''
    for char in text:
        final_text += char
        text_for_display = final_text.replace("<", "&lt;").replace(">", "&gt;")
        container.markdown(f'<div class="info-box">{text_for_display}</div>', unsafe_allow_html=True)
        sleep(delay)
    container.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)
    
def main():
    st.title("🔍 VRAG: Discovering More in Depth")
    st.markdown("""
    <style>
        /* ... CSS 스타일은 이전과 동일 ... */
        .main-title {
            font-size: 36px;
            color: #2E86C1;
            text-align: center;
            margin-bottom: 20px;
        }
        .header-text {
            font-size: 20px;
            color: #34495E;
            margin-bottom: 10px;
        }
        .sidebar-header {
            font-size: 18px;
            color: #1ABC9C;
            margin-bottom: 15px;
        }
        .submit-button {
            background-color: #2E86C1;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .divider {
            border-top: 2px solid #1ABC9C;
            margin-top: 5px;
            margin-bottom: 5px;
        }
        .success-message {
            background-color: #D5F5E3;
            color: #1E8449;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
        .info-box {
            font-size: 18px;
            background-color: #E3F2FD;
            padding: 15px;
            border-left: 6px solid #2E86C1;
            margin-bottom: 10px;
            color: #1E2A38;
        }
        .small-image img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            max-width: auto;
            max-height: 500px;
            text-align: center;
        }
        .caption {
            font-size: 16px;
            color: black;
            text-align: left;
            font-weight: bold;
            margin-left: 10px;
            margin-top: 10px;
            margin-bottom: -5px;
        }
    </style>
    """, unsafe_allow_html=True)

    agent = VRAG(
        planner_model_path="/root/workspace/VRAG_test/VRAG_lsm/grpo_model/30_step_checkpoint"
    )

    with st.sidebar:
        st.markdown('<p class="sidebar-header">⚙️ Configuration Options</p>', unsafe_allow_html=True)
        MAX_ROUNDS = st.number_input('Number of Max Reasoning Iterations:', min_value=3, max_value=10, value=10, step=1)
        
        # ⬅️ [변경] ID와 request_idx 입력을 위한 UI 위젯 추가
        st.markdown('<hr style="border:1px solid #1ABC9C">', unsafe_allow_html=True)
        session_id = st.text_input("Session ID:", value=str(uuid.uuid4()))
        request_idx = st.number_input("Request Index:", min_value=0, value=0, step=1)
        st.markdown('<hr style="border:1px solid #1ABC9C">', unsafe_allow_html=True)
        
        selected_example = st.selectbox(
            'Examples:', 
            ["What is the most commonly used travel app?", 
             "What is the usage rate of travel agents for leisure in the country with the highest credit card penetration rate in Southeast Asia?",
             "助力证券公司上云的核心技术如何保证安全？", 
             "互联网IPV6活跃用户占比在2019年是多少，到2023年增加了多少？"])

    st.markdown('<p class="header-text">📝 Enter Your Question:</p>', unsafe_allow_html=True)

    agent.max_steps = MAX_ROUNDS
    question = st.text_input(
        "Question Input:",
        placeholder="Type your question here...",
        key="question_input",
        value=selected_example,
        label_visibility="collapsed"
    )
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("Submit", key="submit_button", help="Click to submit the question")
    with col2:
        if submit_button and question:
            st.markdown('<p class="success-message">Question submitted! Processing...</p>', unsafe_allow_html=True)

    result_container = st.container()
    with result_container:
        if submit_button and question:
            # ⬅️ [변경] agent.run() 호출 시 session_id와 request_idx 전달
            generator = agent.run(question, session_id=session_id, request_idx=request_idx)
            image_width = 350
            try:
                while True:
                    action, content, raw_content = next(generator)
                    # ... 이하 로직은 이전과 동일 ...
                    if action == 'think':
                        sleep(0.5)
                        think = f"💭 Thinking: {content}"
                    elif action == 'search':
                        typewriter_effect(st.empty(), f'{think} <br> 🔍 <strong>Call Search Engine: {content}</strong>')
                    elif action == 'bbox':
                        bbox_str = content
                        typewriter_effect(st.empty(), f'{think} <br> 📷 <strong>Region of Interest: {content}</strong>')
                    elif action == 'search_image':
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f'<p class="caption">Retrieved Image</p>', unsafe_allow_html=True)
                            st.markdown('<div class="small-image">', unsafe_allow_html=True)
                            st.image(content, width=image_width)
                            st.markdown('</div>', unsafe_allow_html=True)
                    elif action == 'crop_image':
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f'<p class="caption">🤔 Image with bbox: <span style="color:purple;">{bbox_str}</span></p>', unsafe_allow_html=True)
                            st.markdown('<div class="small-image">', unsafe_allow_html=True)
                            st.image(raw_content, width=image_width) 
                            st.markdown('</div>', unsafe_allow_html=True)
                        sleep(0.5)
                        with col2:
                            st.markdown(f'<p class="caption">✂️ Cropped Region</p>', unsafe_allow_html=True)
                            st.markdown('<div class="small-image">', unsafe_allow_html=True)
                            st.image(content, width=image_width) 
                            st.markdown('</div>', unsafe_allow_html=True)
            except StopIteration as e:
                action, content, raw_response = e.value
                if action == 'answer':
                    st.success(f"✅ Answer: {content}")

if __name__ == "__main__":
    main()