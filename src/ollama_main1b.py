import gradio as gr
from document_interface import document_analysis_interface
from chatbot_interface import chatbot_interface
from auth import fetch_user_history  # Import the function for fetching user history
from features_gradio import images,texts

# Define the main app structure
def ollama():
    with gr.Blocks() as app:
        # Tab for document analysis
        with gr.Tab("Analyze documents and ask questions"):
            gr.Markdown("Use the tabs above to navigate through features.")
            document_analysis_interface()

        # Tab for the chatbot
        with gr.Tab("Chat with Chatbot"):
            chatbot_interface()
        
        with gr.Tab("Extract images from pdf"):
            images()
        with gr.Tab("Extract text from images"):
            texts()

        # Tab for user history
        with gr.Tab("User History"):
            gr.Markdown("View your past interactions here.")
            history_output = gr.Textbox(label="History", lines=15, interactive=False)
            fetch_history_button = gr.Button("Fetch History")

            # Connect the button to the function that fetches user history
            fetch_history_button.click(
                fn=fetch_user_history,  # Assuming this function returns the user's history as a string
                inputs=[],
                outputs=history_output
            )

    return app

if __name__ == "__main__":
    app = ollama()
    app.launch(debug=True, share=True)



# # only chatbot and pdf analyze working properly
# import gradio as gr
# from document_interface import document_analysis_interface
# from chatbot_interface import chatbot_interface
# # from chatbot_tavily import chatbot_interface
# # Define the main app structure
# def ollama():
#     with gr.Blocks() as app:
#         # gr.Markdown("## Chat Ollama")
#         with gr.Tab("Analyze documents and ask questions"):
#             gr.Markdown("")
#             # gr.Markdown("Use the tabs above to navigate through features.")
#             document_analysis_interface()
            
#         # Integrate the chatbot tab
#         with gr.Tab("Chat with Chatbot"):
#             chatbot_interface()
#     return app
#     # Launch the app
# if __name__ == "__main__":
#     app=ollama()
#     app.launch(debug=True,share=True)
