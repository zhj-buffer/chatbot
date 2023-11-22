import io
import riva.client
import requests
import soundfile as sf
from functools import partial
import numpy as np
import sys
import queue
import time
import uuid
import json
import argparse
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import cn2an


## Riva ASR 
def dummy_test():
    print("NO THING TO DO")
    
def riva_cn_asr_nonstreaming(asr_url, content):
    if len(content)>0:
        # Set up an offline/batch recognition request
        auth = riva.client.Auth(uri=asr_url) 
        riva_asr = riva.client.ASRService(auth)
        config = riva.client.RecognitionConfig()
        config.encoding = riva.client.AudioEncoding.LINEAR_PCM    # Audio encoding can be detected from wav
        config.sample_rate_hertz = 16000                     # Sample rate can be detected from wav and resampled if needed
        config.language_code = "zh-CN" #"en-US"                    # Language code of the audio clip
        config.max_alternatives = 1                       # How many top-N hypotheses to return
        config.enable_automatic_punctuation = True        # Add punctuation when end of VAD detected
        config.audio_channel_count = 1                    # Mono channel

        response = riva_asr.offline_recognize(content, config)
        asr_best_transcript = response.results[0].alternatives[0].transcript
        print("ASR Transcript:", asr_best_transcript)
        print("\n\nFull Response Message:")
        print(response)
    else:
        asr_best_transcript = 'Riva没有听到您在讲话。'
    return asr_best_transcript

## ChatGLM
def get_chatglm_multi_dialog(chatglm_url, prompt, language='zh', session_id=''):
 
    response_txt = ''
    error_logs = ''
    try:
        json_data = {
            'prompt': prompt,
            'history': []
        }
        response = requests.post(chatglm_url, json=json_data)
        # print(response)
        response_txt = eval(response.text)["response"]
        print(response_txt)
    except Exception as e:
        error_logs = "ChatGLM API response error: "  + str(e)
        print(error_logs)
    if response_txt == '':
        if language =='zh':
            response_txt = '对话机器人没有回答。请检测网络。'
        else:
            response_txt = 'Chatbot did not answer, please check the internet connection.'
    return response_txt, error_logs

## CISI TTS streaming
class UserData:
    def __init__(self, start_request_time):
        self._completed_requests = queue.Queue()
        self._received_length = 0
        self._chunk_count = 0
        self._complete = False
        self._first_chunk_time = None
        self._last_chunk_time = None
        self._start_request_time = start_request_time


def callback(user_data, result, error):
    if error:
        print(error)
        sys.exit(1)
    else:
        if user_data._first_chunk_time is None:
            user_data._first_chunk_time = time.time()

        # Get the waveform chunk
        output_waveform_chunk = result.as_numpy("OUTPUT")[0]
        # Get the valid of the waveform chunk
        output_waveform_length = result.as_numpy("OUTPUT_LENGTH")[0][0]
        # Get the complete flag of the waveform
        complete_flag = result.as_numpy("COMPLETE_FLAG")[0][0]
        # Put the waveform chunk into queue
        user_data._completed_requests.put(output_waveform_chunk)
        # Update the total received waveform length
        user_data._received_length += output_waveform_length
        user_data._chunk_count += 1

        print("Chunk: {}, Received {} samples.".format(user_data._chunk_count,
                                                       output_waveform_length))

        if complete_flag == 1:
            user_data._complete = True
            user_data._last_chunk_time = time.time()
            print("Complete. Received {} chunks.".format(
                user_data._chunk_count))


def append_from_string(string_list: list, input_buffer):
    input_data = []
    for string in string_list:
        input_data.append(string)
    input_data_numpy = np.array(input_data, dtype=object)
    input_data_numpy = input_data_numpy.reshape((len(string_list), 1))
    input_buffer.set_data_from_numpy(input_data_numpy)


def cisiTTS_streaming(text, streaming_url):
    model_name = "tts_scheduler"
    user_name = "user1"
    sentences = text.split() #[text]
    print('###sentences:', sentences)
    uuids = [user_name + "-" + str(uuid.uuid4()) for i in range(len(sentences))]
    uuid_sentence_pairs = zip(uuids, sentences)


    control = {}
    # pitch types: default, flatten, invert, amplify, shift, increase, decrease, etc.
    # you can add more custom types in the model.py of tts frontend.
    pitch_type = 'default'
    pitch_params = {
        "amplify_factor": 1.0,  # only required if pitch_type is amplify
        "shift_value": 0.0  # only required if pitch_type is shift
    }

    # control is a dict json so feel free to add more custom inputs as you need
    control["speaker"] = 'SSB1837'
    control["pace"] = 1.0
    control["pitch_type"] = pitch_type
    control["pitch_params"] = pitch_params
    control_json_str = json.dumps(control)
    index = 0
    start_request_time = time.time()

    waveform = np.array([])

    for uid, sentence in uuid_sentence_pairs:
        index += 1

        inputs = []
        inputs.append(grpcclient.InferInput("INPUT", [1, 1], "BYTES"))
        append_from_string([sentence], inputs[-1])
        inputs.append(grpcclient.InferInput("UUID", [1, 1], "BYTES"))
        append_from_string([uid], inputs[-1])
        inputs.append(grpcclient.InferInput("CONTROL", [1, 1], "BYTES"))
        append_from_string([control_json_str], inputs[-1])

        outputs = []
        outputs.append(grpcclient.InferRequestedOutput("OUTPUT"))
        outputs.append(grpcclient.InferRequestedOutput("OUTPUT_LENGTH"))
        outputs.append(grpcclient.InferRequestedOutput("COMPLETE_FLAG"))

        while time.time() - start_request_time < 0.9999:
            time.sleep(0.0001)

        start_request_time = time.time()

        user_data = UserData(start_request_time)

        with grpcclient.InferenceServerClient(
                url=streaming_url, verbose=False) as triton_client:
            
            try:
                # Establish stream
                triton_client.start_stream(
                    callback=partial(callback, user_data))
                # Send a single inference request
                triton_client.async_stream_infer(model_name=model_name,
                                                 inputs=inputs,
                                                 outputs=outputs)

            except InferenceServerException as error:
                print(error)
                sys.exit(1)

        while user_data._complete == False:
            time.sleep(0.001)
        print('Streaming tts finished.')
        for i in range(user_data._chunk_count):
            waveform_chunk = user_data._completed_requests.get()
            waveform = np.concatenate((waveform, waveform_chunk))

    return {"audio": waveform}  
   


