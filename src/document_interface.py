import gradio as gr
from process_documents import process_youtube,process_csv,process_pdf

# Functions to handle different types of uploads (these can be expanded or modified as needed)
def handle_pdf(file):
    # Your logic for handling PDF uploads
    return f"PDF file {file.name} uploaded successfully!"

def handle_csv(file):
    # Your logic for handling CSV uploads
    return f"CSV file {file.name} uploaded successfully!"

def handle_youtube(link):
    # Your logic for handling YouTube link
    return f"YouTube link '{link}' processed successfully!"



# Function to handle user interaction based on the chosen input type
def show_input(selected_option):
    if selected_option == "PDF":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    elif selected_option == "CSV":
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    elif selected_option == "YouTube":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

def document_analysis_interface():
    with gr.Blocks() as app:
        gr.Markdown("# Document Analysis Interface")

        # Radio buttons for selecting input type
        input_type = gr.Radio(
            choices=["PDF", "CSV", "YouTube"],
            label="Select the type of input"
        )

        # PDF upload section
        with gr.Column(visible=False) as pdf_upload_section:
            pdf_file = gr.File(label="Upload PDF")
            pdf_query = gr.Textbox(label="Enter your question", placeholder="Ask a question about the PDF")
            analyze_pdf_btn = gr.Button("Analyze PDF")
            pdf_output = gr.Textbox(label="PDF Analysis Result",interactive=False)
            # pdf_time_taken=gr.Textbox(label="Total time taken")
            analyze_pdf_btn.click(process_pdf, inputs=[pdf_file, pdf_query], outputs=[pdf_output])

        # CSV upload section
        with gr.Column(visible=False) as csv_upload_section:
            csv_file = gr.File(label="Upload CSV")
            csv_query = gr.Textbox(label="Enter your question", placeholder="Ask a question about the CSV")
            analyze_csv_btn = gr.Button("Analyze CSV")
            csv_output = gr.Textbox(label="CSV Analysis Result",interactive=False)
            # csv_time_taken=gr.Textbox(label="Total time taken")
            analyze_csv_btn.click(process_csv, inputs=[csv_file, csv_query], outputs=[csv_output])

        # YouTube link section
        # with gr.Column(visible=False) as youtube_upload_section:
        #     youtube_link = gr.Textbox(label="Enter YouTube Link")
        #     analyze_youtube_btn = gr.Button("Analyze YouTube")
        #     youtube_output = gr.Textbox(label="YouTube Analysis Result")
        #     youtube_time_taken=gr.Textbox(label="Total time taken")
        #     analyze_youtube_btn.click(process_youtube_and_query, inputs=youtube_link, outputs=[youtube_output,youtube_time_taken])
        
        # YouTube link section
        with gr.Column(visible=False) as youtube_upload_section:
            youtube_link = gr.Textbox(label="Enter YouTube Link")
            query = gr.Textbox(label="Enter your question", placeholder="What is this video about?")
            youtube_output = gr.Textbox(label="YouTube Link Result",interactive=False)
            # response_time_text = gr.Textbox(label="Response Time")
            submit_btn = gr.Button("Analyze")

    # Make sure to pass both the YouTube link and query as inputs
            submit_btn.click(
                process_youtube, 
                inputs=[youtube_link, query], 
                outputs=[youtube_output]
            )


        # Show/hide the sections based on the selected input type
        input_type.change(
            show_input,
            inputs=input_type,
            outputs=[pdf_upload_section, csv_upload_section, youtube_upload_section]
        )

    return app

# Launch the app
if __name__ == "__main__":
    app = document_analysis_interface()
    app.launch(debug=True)
