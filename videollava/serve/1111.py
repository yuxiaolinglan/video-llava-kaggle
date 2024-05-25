# import gradio as gr
# import random
# import time





# with gr.Blocks() as demo:
#     with gr.Column(scale=3):
#         video = gr.Video(label="Input Video")
#     chatbot = gr.Chatbot() # 对话框
#     msg = gr.Textbox() # 输入文本框
#     clear = gr.ClearButton([msg, chatbot]) # 清除按钮
#     """
#     用户输入后的回调函数 respond
#     参数
#     message: 用户此次输入的消息
#     history: 历史聊天记录，比如 [["use input 1", "assistant output 1"],["user input 2", "assistant output 2"]]
    
#     返回值：
#     第一个：设置输入框的值，输入后需要重新清空，所以返回空字符串，
#     第二个：最新的聊天记录
#     """

#     def respond(video,inp, chat_history):
#         video = generate(video)
#         bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
#         chat_history.append((inp, bot_message))
#         time.sleep(2)
#         print(bot_message)
#         print(chat_history)
#         return "", chat_history
    
#     # 绑定输入框内的回车键的响应函数
#     msg.submit(fn = respond, inputs = [video,msg, chatbot], 
#                outputs=[msg, chatbot])

# demo.launch(
#     server_name="0.0.0.0",
#     inbrowser=True,
#     share=False,
#     server_port=9872,
#     quiet=True,
# )






import random
import gradio as gr
import time

def generate(text):
    text = text
    print("yes")
    return "yes"

def echo(message, history):

    return generate(message)

with gr.Blocks() as demo:
    # video = gr.Video(label="Input video")
    time.sleep(2)
    gr.ChatInterface(
        echo
    )

demo.queue().launch()