import rclpy
from rclpy.node import Node

from std_msgs.msg import String

#from wheeltec_mic_msg.msg import PcmMsg

import sys, os
 
#from .nvaudio import *
#
#import pyaudio
#import wave
#import soundfile as sf


##########NV audio module###############
import io
#import riva.client
#import requests
#from functools import partial
#import numpy as np
#import queue
#import time
#import uuid
#import json
#import argparse
#import tritonclient.grpc as grpcclient
#from tritonclient.utils import InferenceServerException
#import cn2an
#import math

from std_msgs.msg import String
from rclpy.qos import QoSProfile
from action_msgs.msg import GoalStatusArray

import re

from geometry_msgs.msg import PointStamped, PoseStamped
from geometry_msgs.msg import Twist

from nav_msgs.msg import Odometry
#import openai
import math
import json


import socketio

###this is global define######

#22050 before, now set to 48000
SPEAKER_RATE =  16000 #22050 #48000 

obj_pos = {
    '展台':(1, 0, 0.0),
    '返回点' : (0, 0, 0.0)
}

history= []
#wav_out_path = "/home/nvidia/chatbot/tts_ret.wav"

tts_prompt_file = "/home/mi/chatbot/src/py_nvasrnlptts/py_nvasrnlptts/isaacsim_basic_0725.txt"

global_switch = True

class colors:  # You may need to change color settings
    RED = "\033[31m"
    ENDC = "\033[m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"


#########define over#########


# reset the robot to its initial pose， 重置 in Chinese。
def mini_reset_robot():
    print('=> mini_reset_robot')

 # get robot's position with respect to the world's frame, returns one numpy array of 3 floats corresponding to XYZ coordinates.
def mini_get_robot_position():
    
    mini_stat = minimal_subscriber.get_odom()

    #print('=> mini_get_robot_position ', mini_stat.pose.pose)
    #print('=> odom.pose.pose ', mini_stat.twist.twist)

    return (mini_stat.pose.pose.position.x, mini_stat.pose.pose.position.y,  mini_stat.pose.pose.position.z)
    #return (0,1,1)

# get robot's orientation with respect to the world's frame, returns one numpy array of 4 floats corresponding to scalar-first quaternion [w,x,y,z]
def mini_get_robot_orientation():
    print('=> mini_get_robot_orientation')
    robot_ori = [1.0, 2.0, 3.0]
    return robot_ori

#get current object's coordinate and return mini position value
def mini_get_obj_pos(obj):
    if obj in obj_pos:
        return obj_pos[obj]
    else:
        print(obj, 'NOT find!')
        return (-1.0, -1.0, -1.0)

# set robot's position with respect to the world's frame, takes one numpy arrays of position [x,y,z].
def mini_set_robot_position(pos):
    print('=> mini_set_robot_position => ', pos)

    #obj not found, return
    if pose.position.x == -1:
        return
 
    action_pub.goal_send(pos)

# set robot's orientation with respect to the world's frame, takes one numpy arrays of scalar-first orientation [w,x,y,z].
def mini_set_robot_orientation(ori_x=0, ori_y=0, ori_z=0):
    print('=> mini_set_robot_orientation = > ', ori_x, ori_y, ori_z)


def mini_move_forward(x=0, y=0, z=0): 
    print('=> mini_move_forward to pos => ', x, y, z)
    #action_pub = MinimalPublisher()
    action_pub.vel_send([0.2, 0.0])
    time.sleep(1)
    action_pub.vel_send([0.0, 0.0])

def mini_turn(theta):
    print('=> mini_turn ', theta)
    action_pub.vel_send([0.0, theta]) 

    #finally set to 0
    #action_pub.vel_send([0.0, 0.0])

 
def mini_go_to_obj(obj):
    pos = mini_get_obj_pos(obj)
    if(pos[0] == -1):
        print(colors.RED + f" mini get pos failed! " + colors.ENDC)
    else:
         mini_go_to_xyz(pos[0], pos[1], pos[2])

def mini_go_to(pos:list):
    mini_go_to(pos[0], pos[1], pos[2])

def mini_go_to(x, y, z):
    print(' => mini_go_to',x, y, z)
        

    ps = PointStamped()
    #ps.header.stamp = self.get_clock().now().to_msg()
    ps.header.frame_id = 'map'
    ps.point.x = float(x)
    ps.point.y = float(y)
    ps.point.z = float(z)
    action_pub.point_send(ps)

    pose = PoseStamped()
    pose.header.frame_id = 'map'
    #pose.header.stamp = rospy.Time.now()
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.position.z = float(z) 
    pose.pose.orientation.z = 0.0
    pose.pose.orientation.w = 0.0


## ChatGPT
def get_chatgpt_multi_dialog(mini_sub, question, language='zh'):

    response_txt = ''
    error_logs = ''

    try:
        #response_txt = mini_sub.chatgpt.ask(question)
        #print('promt : = >', minimal_subscriber.chatgpt.chat_history)

        response_txt = minimal_subscriber.chatgpt.ask(question)
    except Exception as e:
        error_logs = "ChatGPT API response error: "  + str(e)
        print(error_logs)
    if response_txt == '':
        if language =='zh':
            response_txt = '对话机器人没有回答。请检测网络。'
        else:
            response_txt = 'Chatbot did not answer, please check the internet connection.'

    return response_txt


def extract_python_code(content):
    
    code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)

    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)

        if full_code.startswith("python3"):
            full_code = full_code[7:]

        return full_code
    else:
        return None
 
#def chatGPT_control(mini_sub, asr_best_transcript):
#    print('\n\n gpt contrl command ->', asr_best_transcript)
#    resp = get_chatgpt_multi_dialog(mini_sub, asr_best_transcript)
#    print('\n\n gpt feedback =>', resp)
#
#    code = extract_python_code(resp)
#
#    if code is not None:
#        print("\n\n gpt control final command => \n", code)
#        exec(code)
#        print("Done!\n")
#
#def nv_asrnlptts(mini_sub, voice_msg):
#    #save audio into file to check whether record OK
#    #sf.write("output.wav", voice_buf, 22050, "PCM_16")
#
#    global global_switch
#    
#    if(not global_switch):
#        print('asr tts action not finished yet, return \n')
#     
#    global_switch = False
#    
#    minimal_publisher = MinimalPublisher()
#    minimal_publisher.topic_send('yes')# will clear point in multi-points tracking that created by chatGPT
#
#    # Configure the service urls here
#    asr_url = '10.19.206.199:50061'
#    tts_url = '10.19.206.199:18001'
#
#    #path = "/mnt/workspace/mini_robot/src/xf_mic_asr_offline/feedback_voice/voice_control.wav"
#    #asr nlp tts
#    #with io.open(path, 'rb') as fh:
#    #    voice_buf = fh.read()  # bytes
#    
#    # ASR
#    #print(">>>>voice_buf type : ", json.dumps(voice_buf).encode("utf-8"))
#    sample_rate = 16384
#    
#    channel = 1
#
#    voice_buf = voice_msg.pcm_buf_single
#    voice_len = voice_msg.length
#
#    #print(">>type ", type(voice_buf), voice_buf);
#    print(">>type ", type(voice_buf));
#
#    print("===>>> receive len, buflen, buf type \n", voice_len, (len(voice_buf)), type(voice_buf.encode()))
#    
#    print("find \0-->>", voice_buf.find('\0'))
#
#    #save raw data
#       
#    #save json file
#    #with open("/mnt/workspace/mini_robot/pcm.json","w") as f:
#    #   json.dump(voice_buf,f)
#    #voice_list2bytes = json.dumps("".join(voice_buf)).encode("utf-8")
#
#    #print(voice_list2bytes)
#    channel = 1
#
#    pcmfile2wav("/home/nvidia/chatbot/raw.pcm", wav_out_path, channel, SPEAKER_RATE)
#    #pcm2wav(voice_buf.encode(), wav_out_path, channel, sample_rate)    
#    #play_audio_from_wav_file(wav_out_path)
#
#    print("len pcm ", len(voice_buf))
#  
#    voice_buf = '' 
#    with io.open(wav_out_path, 'rb') as fh:
#        voice_buf = fh.read()  # bytes   
#    
#    asr_best_transcript = riva_cn_asr_nonstreaming(asr_url, voice_buf)
#
#    #asr_best_transcript = '绕展台转一圈儿再回来。'
#
#    print("asr result %s \n", asr_best_transcript)
#    
#    if(asr_best_transcript == 'Riva没有听到您在讲话'):
#        converted_reply = asr_best_transcript
#        resp = converted_reply
#    #elif(asr_best_transcript.find('小车')>=0):#index return of key word
#        #print('mini control command ,no responsed \n')
#        #processing control command
#        #print('his before glm ===> ', history)
#        #chatGLM_control(chatglm_url, asr_best_transcript, history, language="zh")
#
#    else:
#        chatGPT_control(mini_sub, asr_best_transcript)
#        #resp, history_, err = get_chatglm_multi_dialog(chatglm_url, asr_best_transcript, history, language="zh")
#        # convert arabic numbers to chinese numbers to pronunce.
#        '''
#        converted_reply = cn2an.transform(resp, "an2cn")
#        
#         # CISI TTS
#        result = cisiTTS_streaming(converted_reply, tts_url)
#        history.append([asr_best_transcript, converted_reply])
#
#        # Write to disk
#        sf.write(wav_out_path, result["audio"], SPEAKER_RATE, "PCM_16") #22050
#    
#        #tell asr do not capture voice 
#        minimal_publisher.topic_send('no')
#
#        #import time #avoid asr hear tts said
#        time.sleep(1)
#
#        #play audio
#        play_audio_from_wav_file(wav_out_path)
#        
#        time.sleep(2)
#
#        #import time #avoid asr hear tts said
#        minimal_publisher.topic_send('yes')
#        '''
#    
#    global_switch = True
#

#class chatGPT(Node):
#
#    def __init__(self):
#        super().__init__('chatgpt_manager')
#
#        with open(tts_prompt_file, "r") as f:
#            prompt = f.read()  
#        
#        self.chat_history = [ 
#            #{
#            #    "role": "system",
#            #    "content": prompt
#            #},
#            {
#                "role": "user",
#                "content": "move 1 meter aong postive x axis"
#            },
#            {
#                "role": "assistant",
#                "content": """```python
#                mini_go_to(mini_get_robot_position()[0] +1 , mini_get_robot_position()[1], mini_get_robot_position()[2]])
#                ```
#                This code uses the `go_to()` function to move the robot to a new position that is 1 units larger in x-axis from the current position. It does this by getting the current position of the robot using `get_robot_position()` and then creating a new list with the same Y and Z coordinates, but with the X coordinate increased by 1. The robot will then go to this new position using `go_to()`."""
#            }
#        ]
#
#        openai.api_key = "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"
#        self.OPENAI_MODEL = "gpt-3.5-turbo"
#      
#        self.ask(prompt)
#        #self.get_logger().info('chatGPT init over !')
#        print(colors.GREEN + f"chatGPT ready for AMR! " + colors.ENDC)
# 
#    def ask(self, prompt):
#
#        #self.get_logger().info('ask to gpt: "%s"' % prompt)
#
#        #self.chat_history = self.chat_history[:3] + self.chat_history[-5:]
#        
#        self.chat_history.append(
#            {
#            "role": "user",
#            "content": prompt,
#            })
#        
#
#        completion = openai.ChatCompletion.create(
#        model = self.OPENAI_MODEL,
#        messages = self.chat_history,
#        temperature=0)
#
#        self.chat_history.append(
#        {
#            "role": "assistant",
#            "content": completion.choices[0].message.content,
#        })
#        
#        print('promt ====>>>> len :', len(self.chat_history))
#        if(len(self.chat_history) > 4):
#            self.chat_history = self.chat_history[:3] + self.chat_history[-1:]
#
#        return self.chat_history[-1]["content"]


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, '/nv/tts', 1)
        #self.pub_vel = self.create_publisher(Twist, '/cmd_vel', 1)
        self.pub_vel = self.create_publisher(Twist, '/mi_desktop_48_b0_2d_7b_03_d5/cmd_vel', 1)
        self.pub_goal = self.create_publisher(PoseStamped, '/goal_pose', 1)

        qos = QoSProfile(depth=10)
        self.pub_point = self.create_publisher(PointStamped, '/clicked_point', qos)#用于发布目标点 

        timer_period = 0.5  # seconds
        #self.timer = self.create_timer(timer_period, self.timer_callback)
        #self.tts_work = False

    def timer_callback(self):
        msg = String()
        #self.publisher_.publish(msg)
        #self.get_logger().info('Publishing: "%s"' % msg.data)
        
    def topic_send(self,enable_capture):
        msg = String()
        msg.data = enable_capture
        self.publisher_.publish(msg)
        #self.get_logger().info('Publishing: "%s"' % msg.data)
    
    def vel_send(self, vel): 
        move_cmd = Twist()
        move_cmd.linear.x = vel[0] * 1 
        print(vel[0])
        move_cmd.angular.z = float(vel[1])/180*math.pi #degree => radian

        #https://www.cnblogs.com/shang-slam/p/6891086.html
        angular_speed = 0.3 ##define MAX_SPEED 0.25 in turn_on_robot
        goal_angular = move_cmd.angular.z
        angular_duration = goal_angular/angular_speed
        rate = 50
        sleep_time = 1.0/rate

        ticks = int(angular_duration*rate)

        print('move_cmd =>', move_cmd)

        for t in range(ticks):
            self.pub_vel.publish(move_cmd)
            time.sleep(sleep_time+0.01)
            self.get_logger().info('Publishing: vel "%d","%d"' % (move_cmd.linear.x, move_cmd.angular.z))

    def goal_send(self, target):
        self.pub_goal.publish(target)

    def point_send(self, target):
        self.pub_point.publish(target)

class mini_status_sub(Node):
    def __init__(self):
        super().__init__('mini_sub_subscriber')
        self.sub_odom = self.create_subscription(
            Odometry,
            #'/odom_combined',
            '/mi_desktop_48_b0_2d_7b_03_d5/odom_out',
            self.odom_callback,
            1)
        self.odom = None #Odometry()
        #self.sub_odom  # prevent unused variable warning

    def odom_callback(self, odom):
        self.get_logger().info('==>>sub : odom retrived... ')

        self.odom = odom

    def get_odom(self):
        return self.odom

#到达目标点成功或失败的回调函数，输入参数：[4：成功， 其它：失败]
def pose_callback(goal_msg):

    GoalReached = False
    print("goal status =>", goal_msg.status_list[-1].status)

    if (goal_msg.status_list[-1].status == 4): #成功到达任意目标点
        GoalReached = True
        print('==>>>>Goal reached ! \n')
 
class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
#        self.sub_pcm = self.create_subscription(
#            PcmMsg,
#            '/mic/pcm/deno',
#            self.listener_callback,
#            1)
#        self.sub_pcm  # prevent unused variable warning
#        self.chatgpt = chatGPT()

        self.odom = None
        self.sub_odom = self.create_subscription(
            Odometry,
            #'/odom_combined',
            '/mi_desktop_48_b0_2d_7b_03_d5/odom_out',
            self.odom_callback,
            1)
        qos = QoSProfile(depth=10)
        self.goal_status_sub = self.create_subscription(GoalStatusArray, '/navigate_to_pose/_action/status', pose_callback, qos)#用于订阅是否到达目标点状态

 
#    def listener_callback(self, msg):
#        #self.get_logger().info('I heard: "%s"' % msg.pcm_buf)
#        print("start to save audio...  \n")
#        nv_asrnlptts(self, msg)
#
#        print("save audio Ok \n")

    def odom_callback(self, odom):
        #self.get_logger().info('==>>sub : odom retrived... ')
        self.odom = odom
    
    def get_odom(self):
        return self.odom





def main(args=None):
    rclpy.init(args=args)


    global minimal_subscriber
    minimal_subscriber = MinimalSubscriber()

    global action_pub
    action_pub = MinimalPublisher()

    global GoalReached
    GoalReached = False
    
    #minimal_subscriber.get_odom()

    global sio
    sio = socketio.Client()
    @sio.event
    
    def connect():
    
        print("Connected to server")
    
        sio.emit('message', 'Hello from Python client!')
    
    @sio.event
    
    def disconnect():
    
        print("Disconnected from server")
    
    @sio.on('response')
    
    def handle_response(data):
    
        print('Received from server:', data)
    
    
    @sio.on('robot_command')
    
    def handle_robot_command(data):
    
        print('Received command:', data)
    
        # Act upon the command with the robot's control system.
    
        code = extract_python_code(data)
    
        if code is not None:
            print("\n\n gpt control final command => \n", code)
            exec(code)
            print("Done!\n")


    sio.connect('http://10.19.162.42:8090')
    
    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':

    
    main()
