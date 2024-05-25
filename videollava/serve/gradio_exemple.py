import os, sys
import gradio as gr
from src.gradio_demo import SadTalker

# ... (rest of the code remains unchanged)

def sadtalker_demo(checkpoint_path='checkpoints', config_path='src/config', warpfn=None):

    sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)

    with gr.Blocks(analytics_enabled=False) as sadtalker_interface:
        gr.Markdown("<div align='center'> <h2> 😭 SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023) </span> </h2> \
                    <a style='font-size:18px;color: #efefef' href='https://arxiv.org/abs/2211.12194'>Arxiv</a>       \
                    <a style='font-size:18px;color: #efefef' href='https://sadtalker.github.io'>Homepage</a>        \
                     <a style='font-size:18px;color: #efefef' href='https://github.com/Winfredy/SadTalker'> Github </div>")

        preprocess_type = gr.Radio(['crop', 'resize','full', 'extcrop', 'extfull'], value='full', label='preprocess', info="How to handle input image?")  # Define the preprocess_type variable here

        with gr.Row():
            enhancer = gr.Radio(['gfpgan1.4', 'gfpgan1.3', 'gfpgan1.2','RestoreFormer','None'], value='gfpgan1.4', label='face enhancer')
            up_scale = gr.Number(value=2, label= 'upscale', min_width=0, info = 'if you let enhancer=None, please let up_scale=0' )
        # Channel multiplier for large networks of StyleGAN2. Default: 2.
        with gr.Row():
            is_still_mode = gr.Checkbox(label="Still Mode (fewer head motion, works with preprocess `full`)", elem_id="is_still_mode")# Define the is_still_mode variable here
            face3dvis = gr.Checkbox(label="generate 3d face and 3d landmarks", elem_id="face3dvis")
           
        
        expression_scale = gr.Slider(label="the batch size of facerender", step=0.01, maximum=1.10, minimum=0.5, value=0.85, elem_id="expression_scale")
        with gr.Row():
            input_yaw_list = gr.Dataframe(type='array', datatype='number', col_count=1, label='input_yaw_list(负数向右,正数向左(以你自己的左右为参照))')
            input_pitch_list = gr.Dataframe(type='array', datatype='number', col_count=1, label='input_pitch_list(大概是脑袋鼓起来和收缩,脑袋变形幅度过大可以用)')
            input_roll_list = gr.Dataframe(type='array', datatype='number', col_count=1, label='input_roll_list(让头旋转, 最好是 -1 0 (min和max不要相差超过1,不然头和身体分离就很明显))')

        batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10, value=2, elem_id="batch_size")  # Define the batch_size variable here

        size_of_image = gr.Radio([256, 512], value=256, label='face model resolution', info="use 256/512 model?", elem_id="size_of_image")  # Define the size_of_image variable here

        pose_style = gr.Slider(minimum=0, maximum=46, step=1, label="Pose style", value=0, elem_id="pose_style")  # Define the pose_style variable here
        with gr.Row():                      
            source_image = gr.Image(label="Source image", type="filepath", elem_id="img2img_image", interactive=True)                
            driven_audio = gr.Audio(label="Input audio", type="filepath", elem_id="driven_audio", interactive=True)  # Define the driven_audio variable here   
            submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')  # Define the submit button here    
            # stop = gr.Button('stop session', elem_id="sadtalker_generate", variant='stop')
        gen_video = gr.Video(label="Generated video", format="mp4", elem_id="gen_video", interactive=True, show_download_button=True)  # Define the gen_video variable here
            # ... (rest of the code remains unchanged)

        if warpfn:
            submit.click(
                            fn=warpfn(sad_talker.test),
                            inputs=[source_image,
                                    driven_audio,
                                    preprocess_type,
                                    is_still_mode,
                                    enhancer,
                                    up_scale,
                                    batch_size,
                                    size_of_image,
                                    pose_style,
                                    expression_scale,
                                    face3dvis,
                                    input_yaw_list, 
                                    input_pitch_list, 
                                    input_roll_list,
                                    ],
                            outputs=[gen_video]
                            )
        else:
            submit.click(
                            fn=sad_talker.test,
                            inputs=[source_image,
                                    driven_audio,
                                    preprocess_type,
                                    is_still_mode,
                                    enhancer,
                                    up_scale,
                                    batch_size,
                                    size_of_image,
                                    pose_style,
                                    expression_scale,
                                    face3dvis,
                                    input_yaw_list, 
                                    input_pitch_list, 
                                    input_roll_list,
                                    ],
                            outputs=[gen_video]
                            )

    return sadtalker_interface


if __name__ == "__main__":

    demo = sadtalker_demo()
    demo.queue().launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=False,
        server_port=9874,
        quiet=True,
    )



from glob import glob
import shutil
import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

def main(args):
    #torch.backends.cudnn.enabled = False

    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style
    device = args.device
    batch_size = args.batch_size
    input_yaw_list = args.input_yaw
    input_pitch_list = args.input_pitch
    input_roll_list = args.input_roll
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose

    current_root_path = os.path.split(sys.argv[0])[0]

    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), args.size, args.old_version, args.preprocess)

    #init model
    preprocess_model = CropAndExtract(sadtalker_paths, device)

    audio_to_coeff = Audio2Coeff(sadtalker_paths,  device)
    
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, args.preprocess,\
                                                                             source_image_flag=True, pic_size=args.size)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, args.preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path=None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ =  preprocess_model.generate(ref_pose, ref_pose_frame_dir, args.preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path=None

    #audio2ceoff
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    # 3dface render
    if args.face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))
    
    #coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess, size=args.size)
    
    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                enhancer=args.enhancer, background_enhancer=args.background_enhancer, preprocess=args.preprocess, img_size=args.size)
    
    shutil.move(result, save_dir+'.mp4')
    print('The generated video is named:', save_dir+'.mp4')

    if not args.verbose:
        shutil.rmtree(save_dir)

    
if __name__ == '__main__':

    parser = ArgumentParser()  
    parser.add_argument("--driven_audio", default='./examples/driven_audio/bus_chinese.wav', help="path to driven audio")
    parser.add_argument("--source_image", default='./examples/source_image/full_body_1.png', help="path to source image")
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2,  help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256,  help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    parser.add_argument('--enhancer',  type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]")
    parser.add_argument('--background_enhancer',  type=str, default=None, help="background enhancer, [realesrgan]")
    parser.add_argument("--cpu", dest="cpu", action="store_true") 
    parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks") 
    parser.add_argument("--still", action="store_true", help="can crop back to the original videos for the full body aniamtion") 
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images" ) 
    parser.add_argument("--verbose",action="store_true", help="saving the intermedia output or not" ) 
    parser.add_argument("--old_version",action="store_true", help="use the pth other than safetensor version" ) 


    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='useless')
    parser.add_argument('--init_path', type=str, default=None, help='Useless')
    parser.add_argument('--use_last_fc',default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # default renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"

    main(args)

