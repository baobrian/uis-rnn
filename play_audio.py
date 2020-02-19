# -*- coding: utf-8 -*-
import pyaudio
import wave

chunk = 1024

wf = wave.open(r".\.wav", 'rb')

p = pyaudio.PyAudio()

# 打开声音输出流
stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = wf.getframerate(),
                output = True)
#读取数据
data = wf.readframes(chunk)
# 写声音输出流进行播放
while data !=b'':
    stream.write(data)
    data = wf.readframes(chunk)
    print(...)

#停止数据流
stream.stop_stream()
stream.close()
p.terminate()
print('播放结束')