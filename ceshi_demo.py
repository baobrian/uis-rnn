# coding=utf-8
import os
import numpy as np
from pyaudio import PyAudio
import wave
import pandas as pd
import math
import uisrnn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tempfile

SAVED_MODEL_NAME = 'test_recognition_model.uisrnn'

# 取真实数据，将Wave音频格式的数据转换成可以处理的数字化数据

def process_wavedata(filepath=None,window_wide=2000):
    # os.chdir(r'E:\uis-rnn\records\\')
    os.chdir(filepath)
    sound_files=os.listdir(filepath)
    name_index=[]
    aliass=['A','B','C','D','E','F','G','H','I','J']
    result=pd.DataFrame()
    # sound_martrix=np.empty(shape=(len(sound_files),220160),dtype=np.short)
    for i,file in enumerate(sound_files):
        name_index.append(file.replace('.wav','')[0:18])
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
        #wave_data.shape = -1, 2
        # 将数组转置
        # wave_data = wave_data.T
        # temp=wave_data[0]
        temp=wave_data.T
        window_length=math.floor((len(temp)-window_wide)/(window_wide/2))
        inner_martrix = np.empty(shape=(window_length, window_wide))
        for j in range(0,window_length):
            print((j*(window_wide/2)),(j*(window_wide/2)+window_wide))
            inner_martrix[j]=temp[int((j*(window_wide/2))):int((j*(window_wide/2)+window_wide))]
        df=pd.DataFrame(data=inner_martrix)
        df['speaker_key']=name_index[i]
        df['group']=aliass[i]
        result=result.append(df)
        result['rank'] = np.arange(0, result.shape[0])
        result.set_index(['rank'], inplace=True)
    result.to_csv(r'E:\Data_temp\records\ceshi_result.csv',header=True,index=False)
    return result


def collect_data(filepath=None):
    os.chdir(filepath)
    return












def train_model():


    return




def predict():

    return




def main():
    model_args, training_args, inference_args = uisrnn.parse_arguments()
    # df=process_wavedata(filepath=r'E:\Data_temp\records\20200225111100\\',window_wide=2000)
    df=pd.read_csv(r'E:\Data_temp\records\ceshi_result.csv')
    df=shuffle(df)

    # train_data.to_csv('audio_tran.csv')
    train_data = df.drop(columns=['speaker_key','group'])
    train_cluster_id= df['speaker_key']
    X_train, X_test, Y_train, Y_test=train_test_split(train_data,train_cluster_id,test_size=0.15, random_state=42)
    train_cluster_id = Y_train.values
    train_sequence = X_train.values
    test_sequence = X_test.values
    test_cluster_id = Y_test.tolist()

    train_sequences=[
        train_sequence[:200,:],
        train_sequence[200:400, :],
        train_sequence[400:700, :],
        train_sequence[700:, :]
    ]
    train_cluster_ids=[
        train_cluster_id[:200],
        train_cluster_id[200:400],
        train_cluster_id[400:700],
        train_cluster_id[700:],
    ]
    # construct model
    model_args, training_args, inference_args = uisrnn.parse_arguments()
    model_args.enable_cuda = True
    model_args.rnn_depth = 2
    model_args.rnn_hidden_size = 8
    model_args.observation_dim = 2000
    model_args.verbosity = 3
    training_args.learning_rate = 0.01
    training_args.train_iteration = 200
    training_args.enforce_cluster_id_uniqueness = False
    inference_args.test_iteration = 2

    model = uisrnn.UISRNN(model_args)

    # run training, and save the model
    model.fit(train_sequences, train_cluster_ids, training_args)
    temp_file_path = tempfile.mktemp()
    model.save(temp_file_path)

    # run testing
    predicted_label = model.predict(test_sequence, inference_args)

    # run evaluation
    model.logger.print(
        3, 'Asserting the equivalence between'
        '\nGround truth: {}\nPredicted: {}'.format(
            test_cluster_id, predicted_label))
    accuracy = uisrnn.compute_sequence_match_accuracy(
        predicted_label, test_cluster_id)



    # model = uisrnn.UISRNN(model_args)
    #
    # # Training.
    # # If we have saved a mode previously, we can also skip training by
    # # calling：
    # # model.load(SAVED_MODEL_NAME)
    # model.fit(train_sequences, train_cluster_ids, training_args)
    # model.save(SAVED_MODEL_NAME)
    # predicted_cluster_ids = []
    # test_record = []
    #
    # for (test_sequence, test_cluster_id) in zip(test_sequence, test_cluster_id):
    #     predicted_cluster_id = model.predict(test_sequence, inference_args)
    #     predicted_cluster_ids.append(predicted_cluster_id)
    #     accuracy = uisrnn.compute_sequence_match_accuracy(
    #         test_cluster_id, predicted_cluster_id)
    #     test_record.append((accuracy, len(test_cluster_id)))
    #     print('Ground truth labels:')
    #     print(test_cluster_id)
    #     print('Predicted labels:')
    #     print(predicted_cluster_id)
    #     print('-' * 100)
    #
    # output_string = uisrnn.output_result(model_args, training_args, test_record)
    #
    # print('Finished diarization experiment')
    # print(output_string)


if __name__ == '__main__':
    main()
