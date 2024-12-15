import re
import gradio as gr
from auth import *  # Assuming auth.py has `handle_login` and `register_user` functions
from model_selection import main
from tables_creation import user_table, history_table
import time


# Password strength checker
def is_password_strong(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter."
    if not re.search(r"\d", password):
        return False, "Password must contain at least one digit."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character."
    return True, ""

# Username validator
def is_username_valid(username):
    if not username.strip():
        return False, "Username cannot be blank."
    if username.isdigit():
        return False, "Username cannot contain only numbers."
    if not re.match(r"^[A-Za-z0-9_]+$", username):
        return False, "Username can only contain letters, numbers, and underscores."
    return True, ""

    # Enhanced login handler with validations
# def enhanced_login(username, password):
#         # Username validation
#         valid_username, username_msg = is_username_valid(username)
#         if not valid_username:
#             return username_msg, gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
        
#         # Blank password check
#         if not password.strip():
#             return "Password cannot be blank.", gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
        
#         # Call handle_login logic
#         success_message = handle_login(username, password,login_tab,register_tab,main_app_row)
#         if success_message == "Login successful":  # Assuming this message indicates success
#             return (
#                 success_message,
#                 gr.update(visible=False),  # Hide login tab
#                 gr.update(visible=False),  # Hide register tab
#                 gr.update(visible=True)    # Show main application
#             )
#         else:
#             return success_message, gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)

# Enhanced registration handler with validations
def enhanced_register(username, password):
    # Username validation
    valid_username, username_msg = is_username_valid(username)
    if not valid_username:
        return username_msg
    # Password strength validation
    strong_password, password_msg = is_password_strong(password)
    if not strong_password:
        return password_msg
    # Call the original register_user function
    registration_message = register_user(username, password)
    return registration_message

# CSS styling
css = """
#login_tab, #register_tab {
    max-width: 500px;
    margin: 0 auto;
    padding: 20px;
    background-color: #333;
    border-radius: 10px;
}
#login_button, #register_button {
    background-color: grey;
    width: 100px;
    text-align: center;
    margin-left: 180px;
}
"""

# Gradio app
with gr.Blocks(css=css) as demo:
    gr.Markdown("<center><b><h2>Chatlytics</h2></b></center>")
    with gr.Row():
        with gr.Tabs() as tabs:
            # Login Tab
            with gr.TabItem("Login", elem_id="login_tab") as login_tab:
                gr.Markdown("<center><b><h4>Login here</h4></b></center>")
                login_username = gr.Textbox(label="Username")
                login_password = gr.Textbox(label="Password", type="password")
                login_button = gr.Button("Login", elem_id="login_button")
                login_status = gr.Text(value="", label="Login Status")

            # Register Tab
            with gr.TabItem("Register", elem_id="register_tab") as register_tab:
                gr.Markdown("<center><b><h4>Register here</h4></b></center>")
                register_username = gr.Textbox(label="Username")
                register_password = gr.Textbox(label="Password", type="password")
                register_button = gr.Button("Register", elem_id="register_button")
                register_status = gr.Text(value="", label="Registration Status")

    # Main app (hidden initially)
    with gr.Row(visible=False) as main_app_row:
        with gr.Column():
            main()


    # Hook buttons to enhanced functions
    login_button.click(
    fn=lambda username, password: handle_login(username, password, login_tab, register_tab, main_app_row),
    inputs=[login_username, login_password],
    outputs=[login_status, login_tab, register_tab, main_app_row]
)


    register_button.click(
        fn=enhanced_register,
        inputs=[register_username, register_password],
        outputs=register_status
    )

if __name__ == "__main__":
    user_table()  # Initialize user table
    history_table()  # Initialize history table
    demo.launch(debug=True, share=True)



# import gradio as gr
# from auth import *
# from model_selection import main
# from tables_creation import user_table, history_table
# # from main_demo import main
# css = """
# #login_tab, #register_tab {
#     max-width: 500px;
#     margin: 0 auto; /* Center the container horizontally */
#     padding: 20px;
#     background-color: #333; /* Optional: Add some styling to distinguish */
#     # background-image: url('/bgimage.jpg')
#     border-radius: 10px;
# #login_button , #register_button{
#     background-color:grey;
#     width:100px;
#     align-text:center;
#     margin-left:180px;
# }
# # bg{
#     background-color:blue;
# }
# }
# """
# with gr.Blocks(css=css) as demo:
#     # Row for login and register tabs
#     gr.Markdown("<center><b><h2> Chatlytics</h2></b></center>")
#     with gr.Row():
#         with gr.Tabs() as tabs:
#             # Login Tab
#             with gr.TabItem("Login", elem_id="login_tab") as login_tab:
#                 gr.Markdown("<center><b><h4>Login here<h4/></b></center>") 
#                 login_message = gr.Text(value="Please log in.",label="",visible=False)
#                 login_username = gr.Textbox(label="Username")
#                 login_password = gr.Textbox(label="Password", type="password")
#                 login_button = gr.Button("Login",elem_id="login_button")
#                 login_status = gr.Text(value="", visible=True,label="login status")  # Error message for login attempts

#             # Register Tab
#             with gr.TabItem("Register", elem_id="register_tab") as register_tab:
#                 gr.Markdown("<center><b><h4>Register yourself here</h4></b></center>")
#                 register_message = gr.Text(value="Create a new account.",label="",visible=False)
#                 register_username = gr.Textbox(label="Username")
#                 register_password = gr.Textbox(label="Password", type="password")
#                 register_button = gr.Button("Register",elem_id="register_button")
#                 register_status = gr.Text(value="",label="status")

#     # Row for the main app (initially hidden)
#     with gr.Row(visible=False) as main_app_row:
#         with gr.Column():
#             main()

#     # Hook up buttons to their respective functionalities
#     login_button.click(
#         fn=lambda username, password: handle_login(username, password, login_tab, register_tab, main_app_row),
#         inputs=[login_username, login_password],
#         outputs=[login_status, login_tab, register_tab, main_app_row]
#     )

#     register_button.click(
#         fn=register_user,
#         inputs=[register_username, register_password],
#         outputs=register_status
#     )

# if __name__ == "__main__":
#     user_table()
#     history_table()
#     # fetch_logged_in_user()
    
#     demo.launch(debug=True, share=True)
