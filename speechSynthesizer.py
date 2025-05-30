
import dashscope
from dashscope.audio.asr import *
from dashscope.audio.tts_v2 import *

from openai import OpenAI
import pygame
from datetime import datetime
import time


# 若没有将API Key配置到环境变量中，需将your-api-key替换为自己的API Key
dashscope.api_key = "sk-9c0fea83ac074bcbab481a077d74c83b"

def get_timestamp():
    now = datetime.now()
    formatted_timestamp = now.strftime("[%Y-%m-%d %H:%M:%S.%f]")
    return formatted_timestamp

class speechSynthesizerCallback(ResultCallback):
    _player = None
    _stream = None
    def __init__(self,synthesizer):
        self.synthesizer = synthesizer

    def on_open(self):
        self.file = open("output.mp3", "wb")
        print(get_timestamp() + " websocket is open.")

    def on_complete(self):
        print(get_timestamp() + " speech synthesis task complete successfully.")
        self.synthesizer.SynthesizerComplete = True

    def on_error(self, message: str):
        print(f"speech synthesis task failed, {message}")

    def on_close(self):
        print(get_timestamp() + " websocket is closed.")
        self.file.close()

    def on_event(self, message):
        pass

    def on_data(self, data: bytes) -> None:
        # print(get_timestamp() + " audio result length: " + str(len(data)))
        self.file.write(data)

class SpeechSynthesizerHandler():
    def __init__(self):
        self.SynthesizerComplete = False

    def forward(self, content):
        self.callback = speechSynthesizerCallback(self)
        self.model = "cosyvoice-v1"
        self.voice = "longxiaochun"
        self.synthesizer = SpeechSynthesizer(
            model=self.model,
            voice=self.voice,
            callback=self.callback,
        )
        self.waitSynthesizerComplete = True

        self.synthesizer.call(content)
        print('[Metric] requestId: {}, first package delay ms: {}'.format(
            self.synthesizer.get_last_request_id(),
            self.synthesizer.get_first_package_delay()))
    
    def readSpeech(self):
        while(not self.SynthesizerComplete):
            print("等待生成语音 ...",end="\r")
            time.sleep(0.1)

        print("\n播放 ...")

        # 初始化pygame
        pygame.init()

        # 加载MP3文件
        pygame.mixer.music.load("output.mp3")

        # 播放MP3文件
        pygame.mixer.music.play()

        # 阻止程序退出，直到音乐播放完成
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.quit()  # 显式退出pygame，释放资源
        self.SynthesizerComplete = False
