# Apocalypse：11 種 OR-Tools 人機

11 個 Python 檔案各自包含完整策略與 CP-SAT 模型，可單獨使用，入口皆為 `make_policy()`。

| 編號 | 策略 | 程式碼 | 優先偏好 |
|---|---|---|---|
| 1 | 大量搬運 | [bot_01_hauler.py](bot_01_hauler.py) | 同趟搬運多件有效物資 |
| 2 | 限時調度 | [bot_02_scheduler.py](bot_02_scheduler.py) | 把握時間窗口與限時獎金 |
| 3 | 工程支援 | [bot_03_engineer.py](bot_03_engineer.py) | 恢復供電、交通及工程事件 |
| 4 | 救援優先 | [bot_04_rescuer.py](bot_04_rescuer.py) | 救援民眾與倒地隊友 |
| 5 | 穩健生存 | [bot_05_survivor.py](bot_05_survivor.py) | 健康、補給及路線風險 |
| 6 | 區域經營 | [bot_06_regional.py](bot_06_regional.py) | 集中服務同一區域 |
| 7 | 輕裝快遞 | [bot_07_sprinter.py](bot_07_sprinter.py) | 輕裝與較短來回行程 |
| 8 | 機會收益 | [bot_08_opportunist.py](bot_08_opportunist.py) | 可取得的任務分數與獎金 |
| 9 | 武裝開路 | [bot_09_vanguard.py](bot_09_vanguard.py) | 武器補給與原版自動防衛 |
| 10 | 補位協作 | [bot_10_balancer.py](bot_10_balancer.py) | 依公開隊友意圖補足工作缺口 |
| 11 | 階段預備 | [bot_11_anticipator.py](bot_11_anticipator.py) | 準備零件、燃油與撤離階段切換 |

## 依賴

使用 Python 及原版遊戲的 [apocalypse_sdk.py](https://github.com/Neoxy-6/Apocalypse/blob/main/game/unity/Assets/StreamingAssets/sdk/apocalypse_sdk.py)，並確保 SDK 可被 Python 匯入。

```bash
python -m pip install ortools==9.15.6755 "requests>=2.28"
```

## 使用

以大量搬運為例；其餘策略更換匯入檔名即可。

```python
from bot_01_hauler import make_policy

policy = make_policy()
steps = policy(st)  # st 為原版 apocalypse_sdk.State。
```

- 每名角色、每場比賽建立新的 `policy`，同一場內持續使用該實例。
- 依原版流程每秒呼叫一次，移動中也呼叫，讓緊急處理及撤離判斷有機會執行。
- 回傳行動串列代表提交新計畫，回傳 `None` 代表維持既有計畫；連線、狀態取得及送出指令沿用原版接口。
- 每次 CP-SAT 求解預設最多 0.25 秒；偏好權重位於各檔案的 `PROFILES`。

## 策略設計

名稱代表優先偏好。11 種人機都保有搬運、救援、工程、補給、避險與撤離能力；共用緊急處理邏輯，也會依主線進度調整選擇。策略只使用原版公開狀態與合法行動，不修改遊戲分數、地圖或規則。

各檔案保留相同的 CP-SAT 路線、容量與時間限制模型，搭配各自權重及策略分支。為方便單檔交付，完整模型直接包含在每個檔案中。

對應遊戲來源版本：`c7004b9bbb74ae5aa381bddaeeee70428544d4a7`。
