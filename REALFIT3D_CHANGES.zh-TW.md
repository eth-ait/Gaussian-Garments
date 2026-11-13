# RealFit3D 對 Gaussian Garments 的修改

[English](REALFIT3D_CHANGES.md) | [README](README.md)

## 在試衣流程中的位置

ContourCraft 先算出人體與衣物逐幀移動的 mesh；GG 將這些形狀與訓練好的
衣物外觀模型結合，渲染成圖片。RealFit3D 再把圖片組成網站上的多角度預覽與影片。

```text
人體與衣物軌跡 + UV 模板 + Stage 3 外觀 checkpoint
    → GG 渲染
    → 多角度 PNG 或動畫影格
    → RealFit3D 網頁預覽與影片
```

此 fork 以 [eth-ait/Gaussian-Garments](https://github.com/eth-ait/Gaussian-Garments)
的 `ceb66fc36598da05b3b3391f4562cdc54d2a2ed2` 為基準，
由 RealFit3D 透過 `upstream/Gaussian-Garments` submodule 固定版本。
以下修改包含渲染入口、路徑設定、輸入檢查，以及 S3 完成後的透明度控制器訓練。
S1 重建、S2 registration、S3 外觀訓練算法維持原版。

## 配準與外觀訓練的輸入輸出

Stage 2 仍以各影格的照片為依據，優化固定拓撲的衣物 mesh。
我們將點雲路徑轉為 Open3D 接受的字串，並將預覽圖裁切到有效像素範圍、
轉為 Pillow 接受的 unsigned 8-bit RGB 陣列。這些修改處理資料載入與
診斷圖片輸出，不改變配準的目標式。

Stage 3 讀取配準後的 mesh 序列、烘焙的 AO／normal 貼圖與標定照片，
使用原有 loss 優化外觀網路及 Gaussian 參數。入口現在會採用指定的
`--epochs`（預設 5），並提供 `--num_workers`（預設 8）。
每個 epoch 完成時，同時寫入 `epochN/` 與 `ckpt/`，讓短序列也能產生
渲染入口使用的固定位置 checkpoint。診斷畫面的貼圖會依照片高度縮放，
避免輸入影像較小時發生拼接越界。

## 模擬結果如何變成衣物圖片

每一幀先讀入模擬算出的衣物頂點與人體位置。GG 讓附著於衣物表面的
Gaussians 隨 mesh 變形，再用 Blender 將當前幾何資訊轉成兩張 UV 貼圖：
AO 貼圖描述衣物受到周圍表面遮蔽的程度，normal 貼圖記錄衣物表面的朝向。
訓練好的 AvatarNet 讀取這兩張貼圖和相機資訊，更新衣物外觀。
最後計算哪些 Gaussians 可見，將它們光柵化成像素，再與人體影像合成。

```text
當前幀的 mesh → UV 遮蔽／法線貼圖 → AvatarNet 更新外觀
             → 可見性計算與 Gaussian 光柵化 → 人體與衣物合成圖
```

這個階段使用已訓練好的 Stage 3 checkpoint，每幀只進行推論。

### 只輸出最終畫面

[render_lean.py](render_lean.py) 保留 `inference.py` 的幾何處理與外觀模型，
只渲染網站需要的最終合成圖。原版入口還會另外渲染 raw／style 比較畫面，
並拼成診斷圖片；我們省去這些額外畫面及拼接工作，輸出動畫需要的逐幀圖片。

在此 repo 目錄內，使用 GG 的 Python 環境並設定好資料路徑後執行：

```bash
python render_lean.py \
  --traj_path /path/to/trajectory.pkl \
  --output_path /path/to/new-render-directory \
  --bake-samples 4
```

輸出是一系列影格圖片，接著由 RealFit3D 的預覽 worker 組成 MP4。

### 同一幀產生多角度預覽

[render_turntable.py](render_turntable.py) 選定模擬中的一幀，
固定衣物姿勢，再讓相機繞著人體拍攝。不同角度共用同一份 mesh 與 UV 貼圖，
因此可重用已產生的幾何與烘焙檔案；相機改變後，仍會重新計算外觀、
可見性及光柵化，得到各角度的圖片。

```bash
python render_turntable.py \
  --traj_path /path/to/trajectory.pkl \
  --output_path /path/to/new-turntable-directory \
  --frame_idx 4 --n-angles 12 --bake-samples 4
```

這組 PNG 供網站顯示穿搭的多角度預覽，完整動作影片則在後續產生。

### 烘焙採樣數的用途

Blender 的烘焙步驟把當前 3D 表面的資訊存成 UV 貼圖，供 AvatarNet 使用。
`--bake-samples` 控制這個步驟使用的 Cycles 採樣數。
新增入口預設為 4，原版 `inference.py` 為 16；降低設定會減少烘焙採樣量，
也可能增加 AO 貼圖的雜訊。貼圖維持模型使用的 512 × 512 大小。
這個參數只調整渲染準備工作，不會改變衣物模擬或重新訓練模型。

## 如何指定資料與模型

我們在 [utils/defaults.py](utils/defaults.py) 加入環境變數設定，
讓 RealFit3D 依部署位置指定輸入、衣物資產與輔助模型，不必逐台修改 GG 程式。

| 變數 | 對應內容 |
|---|---|
| `REALFIT3D_GG_DATA_ROOT` | GG 輸入資料 |
| `REALFIT3D_GG_OUTPUT_ROOT` | 重建後的衣物資產與訓練好的外觀 checkpoint |
| `REALFIT3D_GG_AUX_ROOT` | 輔助模型與資料 |

Renderer 依衣物名稱讀取資產目錄中的 `stage1/template_uv.obj`、
`stage1/cameras.json` 與 `stage3/ckpt/net.pt`；
新渲染的圖片則寫入指令指定的 `--output_path`。

## 如何確認模擬軌跡能交給 GG

多件衣物的頂點會放在同一份軌跡裡，`garment_names` 的排列順序決定每段頂點
對應哪件衣物的 UV 模板與外觀模型。`Simulation` 在渲染前檢查名稱是否合法、
預測序列是否為空、每幀頂點總數是否符合模板，以及人體與衣物幀數是否一致。
輸入不相容時，程式會先回報錯誤。

衣物名稱由模擬 worker 明確寫入，取代依軌跡檔名猜測的方式。
資產準備時仍需對齊面／UV 索引與座標慣例。
由於 pickle 會先反序列化再進行上述檢查，檔案須來自可信來源。

## 透明度控制

三個渲染入口現在共用 [opacity_runtime.py](opacity_runtime.py)，在 Gaussian 光柵化前調整 opacity。
`original` 保留原本渲染行為。
`direct` 模式直接將每顆 Gaussian 的原始 opacity 乘上控制值。
`optical_depth` 以 `1 - (1 - alpha)^s` 調整光學厚度。
V3 則把原始 opacity 與正規化後的初始位置送入小型神經網路，
產生八個正值且總和為一的權重。這些權重組成隨控制值單調變化的分段線性函數，
再以 `1 - (1 - alpha)^m(s)` 調整每顆 Gaussian 的 opacity，交給原有光柵化步驟產生畫面。
`s=0` 隱藏衣物，`s=1` 精確保留原本 Gaussian opacity。
同一次渲染的指定衣物共用一個控制值，各自載入對應的 V3 權重。
`--opacity-garments Actor02_upper` 只改上衣，省略此參數則作用於整套。
未指定的衣物保留原始 opacity。

`Doll` 載入 Stage 3 後，更新模板面座標並保存 Gaussian 位置的獨立副本。
V3 用 checkpoint 的中心與尺度正規化這份固定參考座標；模擬會更新實際渲染幾何，
不會移動控制器的參考位置。載入 checkpoint 後才保存，可包含已訓練的 Gaussian 偏移。

訓練時先固定 Stage 3 模型，渲染原始 alpha 圖 A1，再抽取控制值 s，
將 s × A1 當作目標。Controller 調整每顆 Gaussian 的 opacity，
可微分光柵器用它們產生新 alpha 圖；程式比較新圖與目標的前景誤差，
只更新 controller。訓練完成的 controller 會與 Stage 3 checkpoint 一起用於推論。

[train_opacity_v3.py](train_opacity_v3.py) 讀取指定衣物的 Stage 2 mesh、
point cloud、AO／normal 貼圖、相機校正資料與影像尺寸，以及 Stage 3 checkpoint。
預設以第 0–29 幀訓練、第 30–39 幀驗證，最多選八台相機，執行 5,000 次 Adam 更新。
損失為前景加權 alpha L1，加上 `0.001 * mean((p - 1/8)^2)` 正則項。
產出 `controller.pt`、設定、訓練紀錄及驗證指標；已存在的輸出目錄不會被覆寫。

相機讀取器將完整校正字典與 Stage 2 視角清單結合：字典提供焦距、主點、
世界座標到相機座標的變換，以及影像尺寸；清單限定可參與訓練的配準視角。
程式先逐一比較清單與校正資料的尺寸、焦距和姿態，再選出最多八台訓練相機，
用於渲染 alpha 目標與 controller 預測圖。這一步不使用原始 RGB 照片作為訓練目標。

完整校正資料依序取自 `<data_root>/<subject>/<sequence>/cameras.json`、
Stage 2 的完整校正字典，或 `<output_root>/<subject>/stage1/cameras.json`。
校正資料含 `shape: [height, width]` 且有 Stage 2 視角清單時，不需要原始影像目錄。
舊資料缺少 `shape` 時，會讀 PNG／JPEG 取得尺寸；沒有 Stage 2 清單時，
則由具有校正資料的影像目錄決定可用視角。單獨的渲染相機清單無法還原主點。
訓練設定會記錄校正與清單的路徑及 SHA256。設定中的 data 與 auxiliary 根目錄仍須存在。

以下檢查只使用 CPU，不匯入 Torch 或載入 checkpoint：

```bash
python opacity_training_cameras.py \
  --data-dir /data/Actor02_upper/Sequence1 \
  --stage1-dir /assets/Actor02_upper/stage1 \
  --stage2-dir /assets/Actor02_upper/stage2/Sequence1
```

```bash
python train_opacity_v3.py --subject Actor02_upper --sequence Sequence1
python render_turntable.py \
  --traj_path /path/to/trajectory.pkl --output_path /path/to/new-preview \
  --frame_idx 0 --n-angles 12 --opacity-mode v3 --opacity-scale 0.5 \
  --v3-controller Actor02_upper=/assets/Actor02_upper/opacity_controller_v3/controller.pt \
  --v3-controller Actor02_lower=/assets/Actor02_lower/opacity_controller_v3_valid_cameras/controller.pt
```

同樣的透明參數也適用於 `render_lean.py` 與 `inference.py`。
未指定對應路徑時，V3 會在每件衣物目錄下尋找
`opacity_controller_v3/controller.pt` 或
`opacity_controller_v3_valid_cameras/controller.pt`，必須恰好找到一份。
缺少或重複權重、權重內的衣物名稱不符、座標正規化資料無效時，程式會停止。
權重另行安裝，不包含在 Git 中。

控制器保留小組 V3 的 `4 → 32 → 32 → 8` 網路及 checkpoint 欄位。
本次將固定 Actor02 的訓練路徑改為 subject／sequence 參數，加入輸出保護，
並接到 RealFit3D 使用的渲染入口。這項控制改變畫面透明度，不改變布料力學或模擬材質折射。
來源檔案雜湊與 GPU 實測結果記於
[接入紀錄](https://github.com/Realfit3D/realfit3d/blob/main/docs/OPACITY_INTEGRATION_20261020.md)。

## 版本與測試

渲染入口、路徑設定與輸入檢查由 `8a52dc6` 引入。
控制器與訓練資料處理測試可在 Torch 環境執行
`python -m unittest discover -s tests -v`。
`tests/smoke_opacity_render.py` 使用真實權重檢查兩個渲染入口、
原始／V3 端點及半透明圖像。人體擬合與布料模擬另作完整流程驗收；
重新訓練 V3 需要完整 Stage 2 訓練資料。

版本更新方式見 [RealFit3D 開發流程](https://github.com/Realfit3D/realfit3d/blob/main/docs/DEVELOPMENT.md)。
