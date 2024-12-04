
# only chatbot and pdf analyze working properly
import gradio as gr
from document_interface import document_analysis_interface
from chatbot_interface import chatbot_interface
# from chatbot_tavily import chatbot_interface
# Define the main app structure
with gr.Blocks() as app:
    gr.Markdown("Chat Ollama")
    with gr.Tab("Analyze documents and ask questions"):
        gr.Markdown("")
        # gr.Markdown("Use the tabs above to navigate through features.")
        document_analysis_interface()
        

    # Integrate the chatbot tab
    with gr.Tab("Chat with Chatbot"):
        
    # Integrate the document analysis tab
        chatbot_interface()
    
    # process_pdf_files()

    # Placeholder for other tabs
    # with gr.Tab("Other Features"):
    #     gr.Markdown("### Coming Soon: Additional features will be available here!")

# Launch the app
if __name__ == "__main__":
    app.launch(debug=True,share=True)
