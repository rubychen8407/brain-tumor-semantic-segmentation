## brain-tumor-semantic-segmentation

# 實驗流程
 ![image](https://user-images.githubusercontent.com/73534691/116817024-1b380680-ab97-11eb-88a7-174a39dca78c.png)
  
  * 資料輸入──資料增強(Data augmentation)   
訓練資料僅含有1,280張的影像，可預期在訓練資料集相對較小的情況下，可能產生明顯過擬合的結果。因此，採用四種資料增強的方法：
  -  角度旋轉±45度
  -  對50%的圖像作水平鏡面翻轉；
  -  對50%的圖像作左右翻轉；
  -  對每張影像均勻採樣0至100像素做裁減。    
    
受限於GPU及RAM資源，在實驗過程中將影像由增強約30000張縮減至6400張，每張訓練資料集中的原始影像及腫瘤標記隨機使用四種資料增強的方法產生五張經過資料增強方法的影像及標記，作為新的訓練資料集。

  
* 資料前處理(Data preprocessing)
  - 資料正規化(normalization)：在損失函數優化模型的過程中需使用梯度下降，這時候就需要調整輸入的特徵尺度，所以將像素強度除以255讓尺度降為0~1之間。
  - 資料維度再處理：因為每張影像資料大小及像素皆不同，故將所有影像統一為416 x 416的維度。
  
* 模型比較
  - Modified U-Net:U-Net包含收縮路徑(contracting path)與擴張路徑(expanding path)，形成一個N字型的架構。在收縮路徑時，每次在使用卷積提取特徵後加入資料作標準化(batch normalization)的計算，防止在訓練過程中有資料逐漸偏移的情形，最後使用max pooling的方法將特徵圖像(feature map)縮小; 在擴張路徑時，將down-sampling得到的特徵圖像跟擷取特徵作特徵融合，最後使用up-sampling的方法將特徵放大，避免特徵資訊流失過多。
    
    ![image](https://user-images.githubusercontent.com/73534691/116817261-01e38a00-ab98-11eb-80e5-33e2204a6377.png)
  
  - Deeplabv3 plus:
      
    本實驗於Deeplabv3 plus模型架構於Encoder的部分選擇以修正後的Xception模型為基底(MobileNetV2因架構太小不足以預測此次作業的分類問題)。其中，每經過3x3的深度分離卷積(depth-wise separable convolution)加入資料作標準化(batch normalization)及激活函數RELU，並將所有的最大池化層(max pooling)以補零取代(zero-padding)；於Decoder的部分，設定Output stride為16時，當影像縮小至16倍時，做atrous spatial pyramid pooling，將影像透過不同rate的空洞分離卷積(atrous separable convolution)合併後進行point wise convolution，將得到的feature map放大4倍，與原本104 x 104的低階影像合併，再放大4倍得到最終的feature map。

![image](https://user-images.githubusercontent.com/73534691/116817346-74546a00-ab98-11eb-8ed9-d22f4aaf0438.png)

  
* 訓練模型
  - 優化器(optimizer)：選擇使用Adam，可加快訓練的收斂速度並且對學習率自動作調整。
  - 損失函數(loss function)：以訓練資料集的資料分布作分析，可以發現腫瘤區域的資料相對於其他區域的資料呈現資料分布不均勻的情形，故在損失函數的選擇上，選擇適合用於資料分布不均勻的Dice Loss損失函數，將其與適合用於資料分布較平均的Cross entropy做比較。
  
* 預測測試資料集分類準確度並檢視結果:  
此次作業使用的分類指標不採用accuracy的原因是因為影像分類的前景於背景呈現極度不平衡的情形，導致其不具有參考價值，因此採用Precision、Recall及F1-score作為分類表現指標。
  - Precision: 被模型預測為腫瘤的資料中，有多少是真的有腫瘤的區塊
  - Recall:真的有腫瘤的區塊有多少是被模型預測出有腫瘤的區塊。
  - F1-score: 當在Precision及Recall兩者難以取捨的情況下，衍伸出另一個指標為F1-score，其為Precision 與 Recall 調和平均數 
  

# 實驗結論與討論
在訓練的過程中，由於環境設置條件的限制下，更可以發現dice loss function不論使用哪一種模型，其整體收斂的效果是比cross entropy loss function來得好且在使用relu激活函數下的U-Net模型下呈現的訓練效果是較為穩定的。而論模型表現結果，Dice loss function在不同模型間的綜合表現呈現一致性優於cross entropy loss function的效果，而在使用Dice loss function的狀況下，F1-Score由最佳到最差依序為：deeplabv3+ > U-Net(relu) > U-Net(selu)(詳見下表五)，此結果可證實在資料不均衡的情況下dice loss function 為較適合使用的損失函數，根據Shruti Jadon指出，在資料不均衡的狀況下可預期最佳的損失函數為Focal Tversky Loss (Shruti Jadon, 2020)，未來在實驗過程中可嘗試使用此種損失函數增進表現結果。
  
值得一提的是在診斷腫瘤區塊影響到術後腦功能區域是否受到嚴重影響的議題時，更應該著重於Recall指標，然而因Recall指標較佳的U-Net (selu)_ CE模型綜合表現過低，故建議可選擇使用cross entropy loss function的Deeplabv3+模型做分類，搭配dice loss function的Deeplabv3+模型共同做腫瘤分類預測，避免切除腦部的功能區域，影響術後預後情形。

![image](https://user-images.githubusercontent.com/73534691/116817833-eded5780-ab9a-11eb-9556-3e9d71eba05f.png)

此次實驗嚴重受限於硬體設備的限制，導致在訓練過程中呈現耗時且無效率的情形，耗時時間以U-Net使用dice loss function的訓練過程最久，以epoch為8的訓練次數而言，其平均需花費約4.5小時的訓練時間，訓練過程亦時常遇到GPU及RAM資源用盡等因素導致訓練過程至礙難行。另外，在以cross entropy loss function在訓練U-net(relu)及U-net(selu)的過程中，因硬體設備限制導致模型訓練次數不足，使得此兩種模型其收斂效果不盡理想，未來若在硬體設備允許的範圍內應再增盡訓練次數讓模型訓練結果趨於收斂狀態。

* 參考資料
  - Jadon. S. (2020). A survey of loss functions for semantic segmentation. IEEE Member. IOE: arXiv:2006.14822v4  
  - Magadza. T., and Viriri. S. (2021). Deep Learning for Brain Tumor Segmentation: A Survey of State-of-the-Art. Journal of Imaging. IOE: 10.3390/jimaging7020019
  - Kaku. A., Hegde. C. V., Huang. J. DARTS: Denseunet-Based Automatic Rapid Tool For Brain Segmentation. IOE: arXiv:1911.05567v2
  - Nassar. S. E, Azim. M. A. E,. Elnakib. A. (2020). MRI Brain Tumor Segmentation Using Deep Learning. Mansoura Engineering Journal. 45(4)
  - Samson6460. Tf. Segmentation. Retrieve from: https://github.com/samson6460/tf2_Segmentation
  - Depthwise separable convolution. Retrieve from: https://chih-sheng-huang821.medium.com/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-mobilenet-depthwise-separable-convolution-f1ed016b3467
  - 原始競賽網頁來源：https://aidea-web.tw/topic/a0264109-3fc6-49ea-9e4b-4b0a9433f02f

