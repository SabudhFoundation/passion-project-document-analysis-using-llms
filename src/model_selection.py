import gradio as gr
from ollama_main1b import ollama
from ollama_main3b import ollama3
# from openai_main import openai_main_interface
from amazon_gradio import build_gradio_app

# Function to handle content display based on selection
def show_content(selected_option):
    if selected_option == "ollama-1B":
        return (
            gr.update(visible=True),  # Show ollama-1B section
            gr.update(visible=False),  # Hide OpenAI section
            gr.update(visible=False),  # Hide ollama-3B section
            gr.update(visible=True),  # Show Reset button
        )
    elif selected_option == "ollama-3B":
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),  # Show ollama-3B section
            gr.update(visible=True),
        )
    elif selected_option == "OpenAI":
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
        )
    else:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),  # Hide Reset button
        )

# Function to reset to the initial dropdown state
def reset_selection():
    return (
        gr.update(visible=True),  # Show dropdown
        gr.update(visible=False),  # Hide ollama-1B section
        gr.update(visible=False),  # Hide OpenAI section
        gr.update(visible=False),  # Hide ollama-3B section
        gr.update(visible=False),  # Hide Reset button
    )

# Main app layout
def main():
    with gr.Blocks() as app:
        gr.Markdown("## Select your model")

        # Dropdown for selection
        with gr.Row(visible=True, elem_id="selection_row") as selection_row:
            dropdown = gr.Dropdown(
                choices=["Select an option", "ollama-1B", "ollama-3B", "OpenAI"],
                label="Choose a Model",
                elem_id="model_dropdown"
            )

        # Section for ollama-1B
        with gr.Column(visible=False, elem_id="ollama_section") as ollama_section:
            ollama_interface = ollama()

        # Section for OpenAI
        with gr.Column(visible=False, elem_id="openai_section") as openai_section:
            openai_interface=build_gradio_app()

        # Section for ollama-3B
        with gr.Column(visible=False, elem_id="ollama3_section") as ollama3_section:
            ollama3_interface = ollama3()
        # Reset button
        with gr.Row(visible=False, elem_id="reset_button_row"):
            reset_button = gr.Button("Back to Selection")
            reset_button.click(
                fn=reset_selection,
                outputs=[
                    selection_row,
                    ollama_section,
                    openai_section,
                    ollama3_section,
                    reset_button,
                ]
            )

        # Update content based on the dropdown selection
        dropdown.change(
            fn=show_content,
            inputs=dropdown,
            outputs=[
                ollama_section,
                openai_section,
                ollama3_section,
                reset_button,
            ]
        )

        # Add custom CSS for width control
        gr.HTML("""
            <style>
                #model_dropdown {
                    width: 200px;  /* Adjust the width as needed */
                }
            </style>
        """)

    return app

# Launch the app
if __name__ == "__main__":
    app = main()
    app.launch(debug=True, share=True)


