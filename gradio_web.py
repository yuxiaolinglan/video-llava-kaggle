import torch
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import os, sys
import time
import gradio as gr




def generate(video, inp):
    disable_torch_init()
    
    # video = 'videollava/serve/examples/sample_demo_1.mp4'
    # inp = 'Why is this video funny?'

    model_path = 'LanguageBind/Video-LLaVA-7B'
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)

    # print(f"{roles[1]}: {inp}")
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
    
    
    

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print(outputs)
    return outputs


# 构造 gradio

    """
    用户输入后的回调函数 respond
    参数
    message: 用户此次输入的消息
    history: 历史聊天记录，比如 [["use input 1", "assistant output 1"],["user input 2", "assistant output 2"]]
    
    返回值：
    第一个：设置输入框的值，输入后需要重新清空，所以返回空字符串，
    第二个：最新的聊天 记录
    """
    

def respond(message, history, video):
    output = generate(video,message)
    return output

with gr.Blocks() as demo:
    video = gr.Video(label="Input video")
    time.sleep(2)
    gr.ChatInterface(
        respond, additional_inputs=[video]
    )



    # def respond(video,inp, chat_history):
    #     chat_history.append([None, "结果正在生成，请稍等。"])
    #     bot_message = generate(video, inp)
    #     print(bot_message)
    #     chat_history.append([inp, bot_message])
    #     time.sleep(2)
    #     print("history:", chat_history)
    #     return "", chat_history
    
    # # 绑定输入框内的回车键的响应函数
    # msg.submit(fn=respond, 
    #            inputs=[video, msg, chatbot], 
    #            outputs=[msg, chatbot],
    #            show_progress=True)
    
demo.queue().launch(
    server_name="0.0.0.0",
    share=False,
    server_port=9872,
) 

    