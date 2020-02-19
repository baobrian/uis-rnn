# coding=utf-8
import os
import numpy as np
from pyaudio import PyAudio
import wave
import pandas as pd
import math
import uisrnn
from sklearn.model_selection import train_test_split




SAVED_MODEL_NAME = 'speaker_recognition_model.uisrnn'

# 取真实数据，将Wave音频格式的数据转换成可以处理的数字化数据

def process_wavedata(filepath=None,window_wide=256):
    # os.chdir(r'E:\uis-rnn\records\\')
    os.chdir(filepath)
    sound_files=os.listdir(filepath)
    name_index=[]
    result=pd.DataFrame()
    sound_martrix=np.empty(shape=(len(sound_files),220160),dtype=np.short)
    for i,file in enumerate(sound_files):
        name_index.append(file.replace('.wav',''))
        wf = wave.open(file,'rb')
        print('-'*100)
        print(file)
        p = PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        nframes = wf.getnframes()
        framerate = wf.getframerate()
        # 读取完整的帧数据到str_data中，这是一个string类型的数据
        str_data = wf.readframes(nframes)
        wf.close()
        # 将波形数据转换成数组
        wave_data = np.fromstring(str_data, dtype=np.short)
        # 将wave_data数组改为2列，行数自动匹配
        wave_data.shape = -1, 2
        # 将数组转置
        wave_data = wave_data.T
        temp=wave_data[0]
        inner_martrix = np.empty(shape=(math.floor(len(temp)/window_wide), window_wide))
        for j in range(0,math.floor(len(temp)/window_wide)):
            inner_martrix[j]=temp[j*window_wide:(j+1)*window_wide]
        df=pd.DataFrame(data=inner_martrix)
        df['speaker']=name_index[i]+'_'+str(i)
        df['group']=i
        result=result.append(df)
    return result



def train_model():


    return




def predict():

    return




def main():
    model_args, training_args, inference_args = uisrnn.parse_arguments()
    df=process_wavedata(filepath=r'E:\uis-rnn\records\\',window_wide=2560)
    # train_data.to_csv('audio_tran.csv')
    train_data = df.drop(columns=['speaker','group'])
    train_cluster_id = df['speaker']
    X_train, X_test, Y_train, Y_test=train_test_split(train_data,train_cluster_id,test_size=0.33, random_state=42)
    train_cluster_id = Y_train.values
    train_sequence = X_train.values
    test_sequences = [X_test.values]
    test_cluster_ids = [Y_test.tolist()]
    model = uisrnn.UISRNN(model_args)

    # Training.
    # If we have saved a mode previously, we can also skip training by
    # calling：
    # model.load(SAVED_MODEL_NAME)
    model.fit(train_sequence, train_cluster_id, training_args)
    model.save(SAVED_MODEL_NAME)
    predicted_cluster_ids = []
    test_record = []

    for (test_sequence, test_cluster_id) in zip(test_sequences, test_cluster_ids):
        predicted_cluster_id = model.predict(test_sequence, inference_args)
        predicted_cluster_ids.append(predicted_cluster_id)
        accuracy = uisrnn.compute_sequence_match_accuracy(
            test_cluster_id, predicted_cluster_id)
        test_record.append((accuracy, len(test_cluster_id)))
        print('Ground truth labels:')
        print(test_cluster_id)
        print('Predicted labels:')
        print(predicted_cluster_id)
        print('-' * 100)

    output_string = uisrnn.output_result(model_args, training_args, test_record)

    print('Finished diarization experiment')
    print(output_string)


if __name__ == '__main__':
    main()
