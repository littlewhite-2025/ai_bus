# 從 Ultralytics 匯入套件中的 YOLO 類別
#這裡主要用於用於建立YOLO模型物件，可用來進行物件偵測、模型訓練、驗證等等的操作。
from ultralytics import YOLO

# 載入訓練完成的模型
# best.pt 是 YOLO 在訓練過程中表現最佳(Best) 的權重檔案，這個檔案包含了模型的結構和學習到的權重參數，因此可以直接用來辨識新的圖片。
model = YOLO("runs/detect/seat_detector-3/weights/best.pt")

# 使用模型對指定圖片進行物件偵測
model.predict(
    # source為指定要辨識的圖片，也可以改成影片，網路串流、攝影機、資料夾路徑，這裡使用的是本地端的圖片路徑。
    source="D:/test2/assets/pictures/main/main2.png",

    # save=True表示將偵測結果儲存下來，通常會存到runs/detect/predict/，如果已經存在同名檔案，會在後面加上2, 3, 4....以此類推
    save=True,

    # conf 表示信心值門檻
    # 框框上方顯示的是模型對該物件的信心度，只有當信心度大於0.23時才會將該物件顯示在偵測結果中。
    conf=0.23
)
