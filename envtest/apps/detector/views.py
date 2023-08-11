import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import tensorflow

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

##############################################################################
from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from pathlib import Path

basedir = Path(__file__).parent.parent.parent
dt = Blueprint("detector", __name__, template_folder="templates")


@dt.route("/")
def index():
    return "hello"


@dt.route("/detect")
def detect():
    model = Sequential()
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            input_shape=(32, 32, 3),
            activation="relu",
            padding="same",
        )
    )
    model.add(Dropout(rate=0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(rate=0.25))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(rate=0.25))
    model.add(Dense(10, activation="softmax"))
    print("####################################")
    try:
        model.load_weights("./cifarCnnModel.h5")
        print("success")
    except:
        print("error")

    # 讀取測試圖片
    # img = np.array(Image.open('test.png'))
    # img = np.array(Image.open("qq.png"))
    filename = "a.jpg"
    dir_image = str(basedir / "data" / "original" / filename)
    img = np.array(Image.open(dir_image))

    # # 顯示測試圖片
    # plot_image(img)

    # 建立空的3D Numpy陣列，儲存圖片
    data_test = np.empty((1, 3, 32, 32), dtype="uint8")

    # 將圖片的RGB通道分離後儲存
    data_test[0, :, :, :] = [img[:, :, 0], img[:, :, 1], img[:, :, 2]]

    # 將資料轉換為神經網路所需的格式
    data_test = data_test.transpose(0, 2, 3, 1)

    # 將測試資料進行正規化
    data_test_normalize = data_test.astype("float32") / 255.0

    # 進行圖片分類預測
    prediction = model.predict(data_test_normalize)
    # Predicted_Probability = model.predict(data_test_normalize)
    # print(Predicted_Probability)
    label = current_app.config["LABELS"]
    # print("predict:", label[np.argmax(prediction)])
    print("predict:", label[np.argmax(prediction[0])])

    # # 取出前10個預測結果
    # # prediction = prediction[:10]
    # label = current_app.config["LABELS"]
    # print("predict:", label[np.argmax(prediction)])
    return render_template("detector/index.html", ans=label[np.argmax(prediction[0])])
