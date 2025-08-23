# app.py
import gradio as gr
from teacher_assist import create_question, answer_question, explain_concept


def run(mode, grade, subject, user_input, num_q):
    # filter by meta (optional)
    filter = {"subject": subject, "grade": str(grade)}


    if mode == "Questions":
        return create_question(user_input,filter, num_q)
    elif mode == "Answer":
        return answer_question(user_input, filter)
    elif mode == "Explain":
        return explain_concept(user_input, filter)
    else:
        return "Invalid mode selected. Please choose 'Answer', 'Questions', or 'Explain'."

with gr.Blocks() as demo:
    gr.Markdown("# 📚 Teacher's Assistant")
    with gr.Row():
        grade = gr.Dropdown([str(i) for i in range(1,13)], value="4", label="Grade")
        subject = gr.Dropdown(["math","science","sst","english","hindi"], value="science", label="Subject")
        mode = gr.Radio(["Answer","Questions","Explain"], value="Answer", label="Mode")
        num_q = gr.Slider(1,10,step=1,value=5,label="Questions (if mode=Questions)")
    query = gr.Textbox(lines=3, label="Your question / concept")
    out = gr.Markdown()
    go = gr.Button("Run")
    go.click(run, inputs=[mode, grade, subject, query, num_q], outputs=out)

if __name__ == "__main__":
    demo.launch()