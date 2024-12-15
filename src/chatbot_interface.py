import gradio as gr
from langchain_ollama import ChatOllama
from auth import log_user_history
# from user_history import *

# Initialize the LLaMA model
llm = ChatOllama(model="llama3.2:1b")

# Function to process user queries and maintain chat history
def ai_chat(query, chat_history):
    if not query.strip():
        chat_history.append({"role": "assistant", "content": "Please enter a valid question."})
        return "", chat_history

    try:
        response = llm.invoke(query)
        
        # Check if the response has the content attribute
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)  # Fallback to converting the response to a string if needed

        print(f"LLM Response: {response_text}")

        # Check for empty response
        if not response_text.strip():
            response_text = "I'm sorry, but I didn't understand your question. Could you try again?"

        # Append user query and model response to the chat history
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response_text})
        # if current_user_id:  # Ensure that the user ID is available and logged in
        log_user_history(query, response_text)

        return "", chat_history
    except Exception as e:
        print(f"Error: {e}")
        chat_history.append({"role": "assistant", "content": "An error occurred. Please try again."})
        return "", chat_history


# Function to clear the chat history
def clear_chat():
    return []

# Function to set up the chatbot interface
def chatbot_interface():
    # with gr.Tab("Chat with Chatbot"):
        gr.Markdown("## AI Chat Interface\nChat with the AI model and get your questions answered.")
        chatbox = gr.Chatbot(label="Chat History", type="messages")
        query_input = gr.Textbox(show_label=False, placeholder="Type your question here...", lines=1)
        with gr.Row():
            send_button = gr.Button("Submit")
            clear_button = gr.Button("Clear Chat")
        send_button.click(ai_chat, inputs=[query_input, chatbox], outputs=[query_input, chatbox])
        query_input.submit(ai_chat, inputs=[query_input, chatbox], outputs=[query_input, chatbox])
        clear_button.click(clear_chat, outputs=chatbox)

