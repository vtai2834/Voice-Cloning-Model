import os
import torch
import argparse
import gradio as gr
from zipfile import ZipFile
import langid
import requests

# Phần này cho việc fit thời gian (đầu vào là file srt thay vì text):
import pysrt
from pydub import AudioSegment
import tempfile
import math

print("kiểm tra torch có nhận gpu kh: ", torch.cuda.is_available())


parser = argparse.ArgumentParser()
parser.add_argument("--online_checkpoint_url", default="https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip")
parser.add_argument("--share", action='store_true', default=False, help="make link public")
args = parser.parse_args()

# first download the checkpoints from server
# if not os.path.exists('checkpoints/'):
#     print('Downloading OpenVoice checkpoint ...')
#     os.system(f'wget {args.online_checkpoint_url} -O ckpt.zip')
#     print('Extracting OpenVoice checkpoint ...')
#     ZipFile("ckpt.zip").extractall()
# import requests # Nhớ thêm dòng này ở đầu file

# first download the checkpoints from server
if not os.path.exists('checkpoints/'):
    print('Downloading OpenVoice checkpoint ...')
    # Sử dụng requests để tải file
    url = args.online_checkpoint_url
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open("ckpt.zip", 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        print('Download complete.')
        print('Extracting OpenVoice checkpoint ...')
        ZipFile("ckpt.zip").extractall()
        # Xóa file zip sau khi giải nén (tùy chọn)
        os.remove("ckpt.zip") 
    except Exception as e:
        print(f"Error downloading or extracting file: {e}")

# Init EN/ZH baseTTS and ToneConvertor
from OpenVoice import se_extractor
from OpenVoice.api import BaseSpeakerTTS, ToneColorConverter

en_ckpt_base = 'checkpoints/base_speakers/EN'
zh_ckpt_base = 'checkpoints/base_speakers/ZH'
ckpt_converter = 'checkpoints/converter'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Chạy bằng: ", device)

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
zh_base_speaker_tts = BaseSpeakerTTS(f'{zh_ckpt_base}/config.json', device=device)
zh_base_speaker_tts.load_ckpt(f'{zh_ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device, enable_watermark=False)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
en_source_default_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
en_source_style_se = torch.load(f'{en_ckpt_base}/en_style_se.pth').to(device)
zh_source_se = torch.load(f'{zh_ckpt_base}/zh_default_se.pth').to(device)

supported_languages = ['zh', 'en']

# -----
# Thêm cho srt
# def adjust_audio_speed(audio_path, target_duration):
#     """Điều chỉnh tốc độ audio để khớp với thời lượng mong muốn"""
#     print(f"  Đang điều chỉnh tốc độ: {audio_path}, target_duration: {target_duration}")
#     try:
#         audio = AudioSegment.from_file(audio_path)
#         current_duration = len(audio) / 1000.0  # chuyển sang giây
        
#         print(f"  Độ dài hiện tại: {current_duration}s, độ dài mục tiêu: {target_duration}s")
        
#         if current_duration == 0:
#             print(f"  ⚠️ Cảnh báo: Audio duration là 0")
#             return audio_path
            
#         if target_duration <= 0:
#             print(f"  ⚠️ Cảnh báo: Target duration là {target_duration}")
#             return audio_path
        
#         # Tính hệ số tốc độ
#         speed_factor = current_duration / target_duration
#         print(f"  Hệ số tốc độ: {speed_factor}")
        
#         # Giới hạn hệ số tốc độ trong khoảng hợp lý (0.5x đến 3x)
#         speed_factor = max(0.5, min(3.0, speed_factor))
#         print(f"  Hệ số tốc độ sau giới hạn: {speed_factor}")
        
#         if abs(speed_factor - 1.0) < 0.1:  # Cho phép sai số nhỏ
#             print(f"  Không cần điều chỉnh tốc độ")
#             return audio_path
        
#         # Điều chỉnh tốc độ
#         print(f"  Áp dụng speedup với hệ số: {speed_factor}")
#         adjusted_audio = audio.speedup(playback_speed=speed_factor)
        
#         # Lưu file tạm
#         temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#         adjusted_audio.export(temp_file.name, format="wav")
#         print(f"  Đã lưu audio đã điều chỉnh: {temp_file.name}")
#         return temp_file.name
        
#     except Exception as e:
#         print(f"  ❌ Lỗi trong adjust_audio_speed: {e}")
#         import traceback
#         traceback.print_exc()
#         return audio_path

def adjust_audio_speed(audio_path, target_duration):
    """Điều chỉnh tốc độ audio để khớp với thời lượng mong muốn"""
    from pydub import AudioSegment
    import tempfile
    audio = AudioSegment.from_file(audio_path)
    current_duration = len(audio) / 1000.0  # giây
    if current_duration == 0 or target_duration <= 0.05:
        print(f"  ⚠️ Bỏ qua điều chỉnh tốc độ (cur: {current_duration}, target: {target_duration})")
        return audio_path
    speed_factor = current_duration / target_duration
    speed_factor = max(0.5, min(3.0, speed_factor)) # Giới hạn hợp lý
    adjusted_audio = audio.speedup(playback_speed=speed_factor)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    adjusted_audio.export(temp_file.name, format="wav")
    return temp_file.name

# Helper convert SubRipTime to float seconds
def subrip_to_seconds(t):
    return t.hours * 3600 + t.minutes * 60 + t.seconds + t.milliseconds / 1000

def process_srt_file(srt_file_path, style, speaker_wav, tone_color_converter, tts_model, source_se, language):
    """Xử lý file SRT và tạo audio cho từng segment (cực robust)"""
    from pydub import AudioSegment
    import os
    subs = pysrt.open(srt_file_path)
    processed_segments = []
    print(f"Đang xử lý file SRT với {len(subs)} segments")
    target_se, _ = se_extractor.get_se(
        speaker_wav, tone_color_converter, target_dir='processed', max_length=60., vad=True
    )
    for i, sub in enumerate(subs):
        text = sub.text.strip()
        start_sec = subrip_to_seconds(sub.start)
        end_sec = subrip_to_seconds(sub.end)
        duration = end_sec - start_sec
        print(f"Segment {i}: '{text}' {start_sec}->{end_sec}s | duration={duration}s")
        # Bỏ đoạn text ngắn, không phải ký tự rõ ràng
        if not text or len(text) < 3 or text.isspace() or all(not c.isalnum() for c in text):
            print("  ⚠️ Bỏ qua: text quá ngắn/không hợp lệ")
            continue
        # Bỏ đoạn duration không hợp lệ
        if duration < 0.2:
            print("  ⚠️ Bỏ qua: duration quá ngắn")
            continue
        # Ngôn ngữ segment hợp lệ không
        lang_predicted = langid.classify(text)[0].strip()
        if lang_predicted not in supported_languages:
            print(f"  ⚠️ Bỏ qua: ngôn ngữ {lang_predicted} không hỗ trợ -> dùng tiêng anh")
            lang_predicted = 'en'
            # continue
        try:
            # Sinh TTS
            segment_path = f'{output_dir}/segment_{i}.wav'
            tts_model.tts(text, segment_path, speaker=style, language=language)
            if not os.path.exists(segment_path) or os.path.getsize(segment_path) < 5000:
                print(f"  ❌ File TTS fail/rỗng: {segment_path}")
                continue
            # Convert tone
            converted_segment_path = f'{output_dir}/segment_converted_{i}.wav'
            tone_color_converter.convert(
                audio_src_path=segment_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=converted_segment_path,
                message="@MyShell"
            )
            if not os.path.exists(converted_segment_path) or os.path.getsize(converted_segment_path) < 5000:
                print(f"  ❌ File convert fail/rỗng: {converted_segment_path}")
                continue
            # Kiểm tra current_duration thực tế của file convert
            audio = AudioSegment.from_file(converted_segment_path)
            current_duration = len(audio) / 1000.0
            if current_duration < 0.1:
                print(f"  ❌ Audio convert quá ngắn/rỗng: {current_duration}s")
                continue
            # Điều chỉnh tốc độ (nếu phù hợp)
            final_segment_path = adjust_audio_speed(converted_segment_path, duration)
            # Đảm bảo file sau cùng tồn tại và đủ dài (>0.1s & size>5kB)
            if not os.path.exists(final_segment_path) or os.path.getsize(final_segment_path) < 5000:
                print(f"  ❌ File audio final fail / rỗng: {final_segment_path}")
                continue
            audio_final = AudioSegment.from_file(final_segment_path)
            if len(audio_final) < 100: # <0.1s
                print(f"  ❌ Final audio quá ngắn ({len(audio_final)/1000.0}s)")
                continue
            # Fade in/out 20ms cho segment
            audio_final = audio_final.fade_in(20).fade_out(20)
            # Ghi đè lại final file (khi tạo segment tổng sẽ load lại từ này)
            audio_final.export(final_segment_path, format="wav")
            processed_segments.append({
                'start': start_sec,
                'end': end_sec,
                'audio_path': final_segment_path,
                'text': text
            })
            print(f"  ✅ Đã xử lý OK segment {i}")
        except Exception as e:
            print(f"  ❌ Lỗi segment {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    print(f"Đã xử lý xong {len(processed_segments)} segments hợp lệ")
    return processed_segments, target_se

def create_final_audio(segments, total_duration=None):
    """Ghép audio tổng siêu chắc: nối từng segment đúng timeline, tối ưu không để đoạn silence dài đầu file, không mất tiếng."""
    from pydub import AudioSegment
    import os
    if not segments:
        print("[FINAL_AUDIO] Không có segment hợp lệ, không tạo file final.")
        return None
    # Nếu chỉ có 1 segment, dùng luôn
    if len(segments) == 1:
        print("[FINAL_AUDIO] Chỉ có 1 segment, sử dụng trực tiếp.")
        seg_path = segments[0]['audio_path']
        final_path = f'{output_dir}/final_output.wav'
        AudioSegment.from_file(seg_path).export(final_path, format="wav")
        return final_path
    # Ghép nhiều segment đúng timing SRT
    timeline_audio = AudioSegment.empty()
    prev_end = 0
    for i, seg in enumerate(segments):
        seg_audio = AudioSegment.from_file(seg['audio_path'])
        start_ms = int(seg['start'] * 1000)
        print(f"[FINAL_AUDIO] Ghép segment {i}: start_ms={start_ms}, prev_end={prev_end}, len(seg_audio)={len(seg_audio)}ms")
        # Nếu start_ms > prev_end: cần chèn silence giữa hai segment
        if start_ms > prev_end:
            silence = AudioSegment.silent(duration=(start_ms - prev_end))
            timeline_audio += silence
        # Thêm segment này
        timeline_audio += seg_audio
        prev_end = start_ms + len(seg_audio)
    # Đảm bảo tổng duration tối thiểu theo SRT nếu user yêu cầu
    if total_duration is not None:
        total_ms = int(total_duration * 1000)
        if len(timeline_audio) < total_ms:
            print(f"[FINAL_AUDIO] Padding silence tới tổng {total_ms}ms")
            pad = AudioSegment.silent(duration=(total_ms - len(timeline_audio)))
            timeline_audio += pad
    final_path = f'{output_dir}/final_output.wav'
    timeline_audio.export(final_path, format="wav")
    print(f"[FINAL_AUDIO] Đã xuất file tổng: {final_path}, length={len(timeline_audio)/1000.0}s")
    return final_path

# -----

# def predict(prompt, style, audio_file_pth, mic_file_path, use_mic, agree):
#     # initialize a empty info
#     text_hint = ''
#     # agree with the terms
#     if agree == False:
#         text_hint += '[ERROR] Please accept the Terms & Condition!\n'
#         gr.Warning("Please accept the Terms & Condition!")
#         return (
#             text_hint,
#             None,
#             None,
#         )

#     # first detect the input language
#     language_predicted = langid.classify(prompt)[0].strip()  
#     print(f"Detected language:{language_predicted}")

#     if language_predicted not in supported_languages:
#         text_hint += f"[ERROR] The detected language {language_predicted} for your input text is not in our Supported Languages: {supported_languages}\n"
#         gr.Warning(
#             f"The detected language {language_predicted} for your input text is not in our Supported Languages: {supported_languages}"
#         )

#         return (
#             text_hint,
#             None,
#             None,
#         )
    
#     if language_predicted == "zh":
#         tts_model = zh_base_speaker_tts
#         source_se = zh_source_se
#         language = 'Chinese'
#         if style not in ['default']:
#             text_hint += f"[ERROR] The style {style} is not supported for Chinese, which should be in ['default']\n"
#             gr.Warning(f"The style {style} is not supported for Chinese, which should be in ['default']")
#             return (
#                 text_hint,
#                 None,
#                 None,
#             )

#     else:
#         tts_model = en_base_speaker_tts
#         if style == 'default':
#             source_se = en_source_default_se
#         else:
#             source_se = en_source_style_se
#         language = 'English'
#         if style not in ['default', 'whispering', 'shouting', 'excited', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']:
#             text_hint += f"[ERROR] The style {style} is not supported for English, which should be in ['default', 'whispering', 'shouting', 'excited', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']\n"
#             gr.Warning(f"The style {style} is not supported for English, which should be in ['default', 'whispering', 'shouting', 'excited', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']")
#             return (
#                 text_hint,
#                 None,
#                 None,
#             )

#     if use_mic == True:
#         if mic_file_path is not None:
#             speaker_wav = mic_file_path
#         else:
#             text_hint += f"[ERROR] Please record your voice with Microphone, or uncheck Use Microphone to use reference audios\n"
#             gr.Warning(
#                 "Please record your voice with Microphone, or uncheck Use Microphone to use reference audios"
#             )
#             return (
#                 text_hint,
#                 None,
#                 None,
#             )

#     else:
#         speaker_wav = audio_file_pth

#     if len(prompt) < 2:
#         text_hint += f"[ERROR] Please give a longer prompt text \n"
#         gr.Warning("Please give a longer prompt text")
#         return (
#             text_hint,
#             None,
#             None,
#         )
#     # if len(prompt) > 200:
#     #     text_hint += f"[ERROR] Text length limited to 200 characters for this demo, please try shorter text. You can clone our open-source repo and try for your usage \n"
#     #     gr.Warning(
#     #         "Text length limited to 200 characters for this demo, please try shorter text. You can clone our open-source repo for your usage"
#     #     )
#     #     return (
#     #         text_hint,
#     #         None,
#     #         None,
#     #     )
    
#     # note diffusion_conditioning not used on hifigan (default mode), it will be empty but need to pass it to model.inference
#     try:
#         target_se, wavs_folder = se_extractor.get_se(speaker_wav, tone_color_converter, target_dir='processed', max_length=60., vad=True)
#         # os.system(f'rm -rf {wavs_folder}')
#     except Exception as e:
#         text_hint += f"[ERROR] Get target tone color error {str(e)} \n"
#         gr.Warning(
#             "[ERROR] Get target tone color error {str(e)} \n"
#         )
#         return (
#             text_hint,
#             None,
#             None,
#         )

#     src_path = f'{output_dir}/tmp.wav'
#     tts_model.tts(prompt, src_path, speaker=style, language=language)

#     save_path = f'{output_dir}/output.wav'
#     # Run the tone color converter
#     encode_message = "@MyShell"
#     tone_color_converter.convert(
#         audio_src_path=src_path, 
#         src_se=source_se, 
#         tgt_se=target_se, 
#         output_path=save_path,
#         message=encode_message)

#     text_hint += f'''Get response successfully \n'''

#     return (
#         text_hint,
#         save_path,
#         speaker_wav,
#     )


def predict(srt_file, style, audio_file_pth, mic_file_path, use_mic, agree):
    text_hint = ''
    # Kiểm tra có đồng ý điều khoản không
    if agree == False:
        text_hint += '[ERROR] Please accept the Terms & Condition!\n'
        gr.Warning("Please accept the Terms & Condition!")
        return (
            text_hint,
            None,
            None,
        )

    # Kiểm tra đầu vào SRT
    if srt_file is None:
        text_hint += "[ERROR] Vui lòng upload file SRT\n"
        gr.Warning("Vui lòng upload file SRT")
        return text_hint, None, None

    # Lấy đường dẫn file nếu là file-like object (Gradio File type="file")
    srt_path = srt_file.name if hasattr(srt_file, "name") else srt_file

    # Kiểm tra tham số style
    # (Chấp nhận các giá trị style hợp lệ bên dưới khi xử lý từng ngôn ngữ)

    # Kiểm tra chọn mic hay file audio tham chiếu
    if use_mic is True:
        if mic_file_path is not None:
            speaker_wav = mic_file_path
        else:
            text_hint += "[ERROR] Vui lòng ghi âm giọng nói của bạn\n"
            gr.Warning("Vui lòng ghi âm giọng nói của bạn")
            return text_hint, None, None
    else:
        speaker_wav = audio_file_pth

    try:
        # Xác định ngôn ngữ chính từ file SRT (có thể cải tiến để xử lý đa ngôn ngữ)
        subs = pysrt.open(srt_path)
        sample_text = subs[0].text if len(subs) > 0 else ""
        language_predicted = langid.classify(sample_text)[0].strip()
        
        if language_predicted == "zh":
            tts_model = zh_base_speaker_tts
            source_se = zh_source_se
            language = 'Chinese'
            if style not in ['default']:
                text_hint += f"[ERROR] The style {style} is not supported for Chinese, which should be in ['default']\n"
                gr.Warning(f"The style {style} is not supported for Chinese, which should be in ['default']")
                return (
                    text_hint,
                    None,
                    None,
                )
        else:
            tts_model = en_base_speaker_tts
            source_se = en_source_default_se if style == 'default' else en_source_style_se
            language = 'English'
            if style not in ['default', 'whispering', 'shouting', 'excited', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']:
                text_hint += f"[ERROR] The style {style} is not supported for English, which should be in ['default', 'whispering', 'shouting', 'excited', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']\n"
                gr.Warning(f"The style {style} is not supported for English, which should be in ['default', 'whispering', 'shouting', 'excited', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']")
                return (
                    text_hint,
                    None,
                    None,
                )

        # Xử lý file SRT, sinh ra các segment audio và ghép thành file đầu ra
        segments, target_se = process_srt_file(
            srt_path, style, speaker_wav, tone_color_converter, 
            tts_model, source_se, language
        )

        if not segments:
            text_hint += "[ERROR] Không có segment nào được xử lý thành công\n"
            return text_hint, None, None

        # Tạo audio cuối cùng
        final_audio_path = create_final_audio(segments)
        
        text_hint += f"Xử lý thành công {len(segments)} segments\n"
        return text_hint, final_audio_path, speaker_wav
    except Exception as e:
        text_hint += f"[ERROR] Lỗi xử lý: {str(e)}\n"
        return text_hint, None, None


title = "MyShell OpenVoice"

description = """
We introduce OpenVoice, a versatile instant voice cloning approach that requires only a short audio clip from the reference speaker to replicate their voice and generate speech in multiple languages. OpenVoice enables granular control over voice styles, including emotion, accent, rhythm, pauses, and intonation, in addition to replicating the tone color of the reference speaker. OpenVoice also achieves zero-shot cross-lingual voice cloning for languages not included in the massive-speaker training set.
"""

markdown_table = """
<div align="center" style="margin-bottom: 10px;">

|               |               |               |
| :-----------: | :-----------: | :-----------: | 
| **OpenSource Repo** | **Project Page** | **Join the Community** |        
| <div style='text-align: center;'><a style="display:inline-block,align:center" href='https://github.com/myshell-ai/OpenVoice'><img src='https://img.shields.io/github/stars/myshell-ai/OpenVoice?style=social' /></a></div> | [OpenVoice](https://research.myshell.ai/open-voice) | [![Discord](https://img.shields.io/discord/1122227993805336617?color=%239B59B6&label=%20Discord%20)](https://discord.gg/myshell) |

</div>
"""

markdown_table_v2 = """
<div align="center" style="margin-bottom: 2px;">

|               |               |               |              |
| :-----------: | :-----------: | :-----------: | :-----------: | 
| **OpenSource Repo** | <div style='text-align: center;'><a style="display:inline-block,align:center" href='https://github.com/myshell-ai/OpenVoice'><img src='https://img.shields.io/github/stars/myshell-ai/OpenVoice?style=social' /></a></div> |  **Project Page** |  [OpenVoice](https://research.myshell.ai/open-voice) |     

| | |
| :-----------: | :-----------: |
**Join the Community** |   [![Discord](https://img.shields.io/discord/1122227993805336617?color=%239B59B6&label=%20Discord%20)](https://discord.gg/myshell) |

</div>
"""
content = """
<div>
  <strong>For multi-lingual & cross-lingual examples, please refer to <a href='https://github.com/myshell-ai/OpenVoice/blob/main/demo_part2.ipynb'>this jupyter notebook</a>.</strong>
  This online demo mainly supports <strong>English</strong>. The <em>default</em> style also supports <strong>Chinese</strong>. But OpenVoice can adapt to any other language as long as a base speaker is provided.
</div>
"""
wrapped_markdown_content = f"<div style='border: 1px solid #000; padding: 10px;'>{content}</div>"


examples = [
    [
        "今天天气真好，我们一起出去吃饭吧。",
        'default',
        "examples/speaker0.mp3",
        None,
        False,
        True,
    ],[
        "This audio is generated by open voice with a half-performance model.",
        'whispering',
        "examples/speaker1.mp3",
        None,
        False,
        True,
    ],
    [
        "He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered, flour-fattened sauce.",
        'sad',
        "examples/speaker2.mp3",
        None,
        False,
        True,
    ],
]

with gr.Blocks(analytics_enabled=False) as demo:

    with gr.Row():
        with gr.Column():
            with gr.Row():
                gr.Markdown(
                    """
                    ## <img src="https://huggingface.co/spaces/myshell-ai/OpenVoice/raw/main/logo.jpg" height="40"/>
                    """
                )
            with gr.Row():    
                gr.Markdown(markdown_table_v2)
            with gr.Row():
                gr.Markdown(description)
        with gr.Column():
            gr.Video('./open_voice.mp4', autoplay=True)
            
    with gr.Row():
        gr.HTML(wrapped_markdown_content)

    with gr.Row():
        with gr.Column():
            # input_text_gr = gr.Textbox(
            #     label="Text Prompt",
            #     info="You can enter as much text as you wish.",
            #     value="He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered, flour-fattened sauce.",
            # )

            # Thay thế Textbox bằng File upload cho SRT
            srt_file_gr = gr.File(
                label="File SRT",
                info="Upload file phụ đề .srt",
                file_types=[".srt"],
                type="file"
            )

            style_gr = gr.Dropdown(
                label="Style",
                info="Select a style of output audio for the synthesised speech. (Chinese only support 'default' now)",
                choices=['default', 'whispering', 'cheerful', 'terrified', 'angry', 'sad', 'friendly'],
                max_choices=1,
                value="default",
            )
            ref_gr = gr.Audio(
                label="Reference Audio",
                info="Click on the ✎ button to upload your own target speaker audio",
                type="filepath",
                value="examples/speaker0.mp3",
            )
            mic_gr = gr.Audio(
                source="microphone",
                type="filepath",
                info="Use your microphone to record audio",
                label="Use Microphone for Reference",
            )
            use_mic_gr = gr.Checkbox(
                label="Use Microphone",
                value=False,
                info="Notice: Microphone input may not work properly under traffic",
            )
            tos_gr = gr.Checkbox(
                label="Agree",
                value=False,
                info="I agree to the terms of the cc-by-nc-4.0 license-: https://github.com/myshell-ai/OpenVoice/blob/main/LICENSE",
            )

            tts_button = gr.Button("Send", elem_id="send-btn", visible=True)


        with gr.Column():
            out_text_gr = gr.Text(label="Info")
            audio_gr = gr.Audio(label="Synthesised Audio", autoplay=True)
            ref_audio_gr = gr.Audio(label="Reference Audio Used")

            # gr.Examples(examples,
            #             label="Examples",
            #             inputs=[input_text_gr, style_gr, ref_gr, mic_gr, use_mic_gr, tos_gr],
            #             outputs=[out_text_gr, audio_gr, ref_audio_gr],
            #             fn=predict,
            #             cache_examples=False,)
            # tts_button.click(predict, [input_text_gr, style_gr, ref_gr, mic_gr, use_mic_gr, tos_gr], outputs=[out_text_gr, audio_gr, ref_audio_gr])

    # Cập nhật examples để sử dụng SRT
    examples = [
        [
            "examples/subtitles.srt",  # Thay thế bằng file SRT mẫu
            'default',
            "examples/speaker0.mp3",
            None,
            False,
            True,
        ],
    ]

    gr.Examples(
        examples,
        label="Ví dụ",
        inputs=[srt_file_gr, style_gr, ref_gr, mic_gr, use_mic_gr, tos_gr],
        outputs=[out_text_gr, audio_gr, ref_audio_gr],
        fn=predict,
        cache_examples=False,
    )
    
    tts_button.click(predict, [srt_file_gr, style_gr, ref_gr, mic_gr, use_mic_gr, tos_gr], 
                    outputs=[out_text_gr, audio_gr, ref_audio_gr])

demo.queue()  
demo.launch(debug=True, show_api=True, share=args.share)

# demo.queue()  
# demo.launch(debug=True, show_api=True, share=args.share)