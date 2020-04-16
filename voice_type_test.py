from sklearn.svm import SVC
import numpy
from mfcc import *
import glob
from sklearn.metrics import classification_report, accuracy_score
import csv
from sklearn.model_selection import cross_val_score
import argparse
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cross-validation",
                        help="execute cross validation test",
                        action="store_true",
                        default=False)
    args = parser.parse_args()

    files_name_light = glob.glob("learning_sample/light/*.wav")
    files_name_pull = glob.glob("learning_sample/pull/*.wav")

    nfft = 2048  # FFTのサンプル数
    nceps = 12   # MFCCの次元数

    train_label = np.array([])
    test_label = np.array([])

    train_data = np.empty((0, 12), float)
    test_data = np.empty((0, 12), float)

    for file_name_light in files_name_light:
        feature = get_feature(file_name_light, nfft, nceps)
        if len(train_data) == 0:
            train_data = feature
        else:
            train_data = np.vstack((train_data,feature))
        train_label = np.append(train_label,'light')

    for file_name_pull in files_name_pull:
        feature = get_feature(file_name_pull, nfft, nceps)
        if len(train_data) == 0:
            train_data = feature
        else:
            train_data = np.vstack((train_data,feature))
        train_label = np.append(train_label,'pull')

    clf = SVC(kernel='rbf', C=1)
    if args.cross_validation :
        # 交差検証
        stratifiedkfold = StratifiedKFold(n_splits=10)
        scores = cross_val_score(clf, train_data, train_label, cv=stratifiedkfold)
        # 各分割におけるスコア
        print('Cross-Validation scores: {}'.format(scores))
        # スコアの平均値
        import numpy as np
        print('Average score: {}'.format(np.mean(scores)))

    # SVC学習
    clf.fit(train_data, train_label)

    # ここにテストデータを入れる
    test_voices = glob.glob("no_processing_sound/voice/*.wav")
    for test_voice in test_voices:
        test_feature = get_feature(test_voice, nfft, nceps)
        if len(test_data) == 0:
            test_data = np.vstack((test_data, test_feature))
        else:
            test_data = np.vstack((test_data,test_feature))
        test_label_name = test_voice.split('/')[2].split('_')[0]
        test_label = np.append(test_label, test_label_name)

        # 検証
        score = clf.score(test_data, test_label)
        print(score)

    #特徴データをテキストに出力
    # feature_train_data=np.hstack((train_label.reshape(len(train_label),1),train_data))
    # feature_test_data = np.hstack((test_label.reshape(len(test_label), 1), test_data))
    
    # with open("feature_data/train_data.txt","w") as f:
    #     writer=csv.writer(f) # writerオブジェクトを作成
    #     writer.writerows(feature_train_data) # feature_train_dataを書き込む
    # with open("feature_data/train_data.txt") as f:
    #     print(f.read())
        
    # with open("feature_data/test_data.txt","w") as f:
    #     writer=csv.writer(f) # writerオブジェクトを作成
    #     writer.writerows(feature_test_data) # feature_test_dataを書き込む
