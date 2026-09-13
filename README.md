# # LocalVoice/ 实时语音输入助手

纯CPU计算，基于 **Sherpa-Onnx 流式 Paraformer** 的离线实时语音输入法:按住快捷键说话,识别结果经过数字规范化、标点恢复、中英文排版优化后,自动"打字"上屏到任意输入框。全程本地运行,无需联网。

## 一、整体架构

```
麦克风 (sounddevice)
        │  100ms 音频块
        ▼
   音频队列 (queue.Queue)
        │
        ▼
┌─────────────────────────────────────────────┐
│              核心处理管线(三房间)              │
│                                             │
│  房间 1: ASR 识别                            │
│    · Sherpa-Onnx 流式 Paraformer 中英双语模型  │
│    · 流式解码 + 静音断句(Endpoint 检测)        │
│                                             │
│  房间 2: ITN 逆文本正则化                     │
│    · cn2an 纯 Python 数字转换                 │
│    · 例:六百美元 -> 600美元,三点一倍 -> 3.1倍  │
│                                             │
│  房间 3: 标点恢复 + 排版优化                   │
│    · CT-Transformer ONNX 离线标点预测          │
│    · 中文与英文/数字之间自动插入空格            │
└─────────────────────────────────────────────┘
        │
        ▼
剪贴板保护上屏 (pyperclip + pyautogui)
  复制文本 → 模拟 Ctrl+V → 延时恢复原剪贴板内容
```

### 三大核心模块说明

| 模块         | 技术                                 | 作用                                  |
| ---------- | ---------------------------------- | ----------------------------------- |
| 房间 1:语音识别  | sherpa-onnx 流式 Paraformer(int8 量化) | 实时把语音转成文字,支持中英混合,静音 1.2~2.4 秒自动断句定稿 |
| 房间 2:数字规范化 | cn2an                              | 把口语中文数字转成阿拉伯数字(免去 pynini 编译陷阱)      |
| 房间 3:标点与排版 | CT-Transformer ONNX + 正则           | 为无标点文本预测逗号/句号,并在中英文/数字边界插入空格        |

### 线程模型

- **音频回调线程**:sounddevice 回调采集音频,仅在监听状态入队。
- **识别工作线程** `_process_audio_loop`:从队列取音频喂给流式识别器,终端实时预览未定稿文本,Endpoint 触发时整句定稿并上屏。
- **热键监听线程**:pynput 全局监听 `Ctrl+Shift+Q` 切换开始/停止。
- **剪贴板恢复线程**:上屏后延时 0.3 秒恢复用户原有剪贴板内容。

## 二、环境依赖

- Python 3.9+
- Windows / macOS / Linux
- 依赖库:

```bash
pip install --upgrade sherpa-onnx sounddevice pyperclip pyautogui pynput cn2an
```

## 三、模型准备

### 项目目录结构

```
实时语音输入/
├── realtime_typing_v5.py                                # 主脚本
├── hotwords.txt                                         # 热词文件(可选,当前版本不生效)
├── sherpa-onnx-streaming-paraformer-bilingual-zh-en/    # ASR 模型目录
│   ├── tokens.txt                                       # 词表(必需)
│   ├── encoder.int8.onnx                                # 编码器(必需)
│   └── decoder.int8.onnx                                # 解码器(必需)
└── sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/  # 标点模型目录
    └── model.onnx                                       # 标点模型(必需,约 281MB)
```

> 说明:标点模型压缩包解压后还包含 `tokens.json`、`config.yaml`、`test.py` 等文件,运行脚本只需 `model.onnx`,其余可删除;ASR 目录三个文件缺一不可,启动时缺失会直接报错退出。

在脚本所在目录放置两个模型目录:

1. **流式 ASR 模型**(必需):`sherpa-onnx-streaming-paraformer-bilingual-zh-en`
   - 需包含 `tokens.txt`、`encoder.int8.onnx`、`decoder.int8.onnx`
   - 下载地址: [csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en at main](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en/tree/main)
2. **标点模型**(可选):`sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12`
   - 需包含 `model.onnx`
   - 下载地址: [csukuangfj/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12 at main](https://huggingface.co/csukuangfj/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/tree/main)
3. **热词文件**(可选):`hotwords.txt`
   - ⚠️ 当前 sherpa-onnx 1.13.x 的 `from_paraformer()` 已移除热词文件参数,脚本会忽略此文件并打印警告

## 四、使用方法

```bash
cd D:\AI_Tools\实时语音输入
python realtime_typing_v5.py
```

启动后:

1. 等待日志出现 `系统初始化完成!`
2. 按 **Ctrl + Shift + Q** 开始语音输入(日志显示 🎙️)
3. 对着麦克风说话,终端会实时显示 `[实时识别] ...` 预览
4. 说话停顿约 1~2 秒自动断句,整句自动上屏到当前光标位置
5. 再按 **Ctrl + Shift + Q** 停止监听(当前未定稿内容会立即定稿上屏)
6. 终端按 **Ctrl + C** 退出程序

## 五、配置项 (AppConfig)

| 配置                | 默认值                                                               | 说明             |
| ----------------- | ----------------------------------------------------------------- | -------------- |
| `sample_rate`     | 16000                                                             | 采样率,需与模型一致     |
| `block_size`      | 1600                                                              | 音频块大小(100ms)   |
| `asr_model_dir`   | `./sherpa-onnx-streaming-paraformer-bilingual-zh-en`              | ASR 模型目录       |
| `punct_model_dir` | `./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12` | 标点模型目录         |
| `hotwords_file`   | `./hotwords.txt`                                                  | 热词文件(当前版本暂不支持) |
| `num_threads`     | 4                                                                 | ASR 推理线程数      |
| `paste_delay`     | 0.05                                                              | 复制到粘贴之间的延时(秒)  |
| `hotkey_str`      | `<ctrl>+<shift>+q`                                                | 语音开关快捷键        |
| `enable_itn`      | True                                                              | 启用 cn2an 数字转换  |
| `enable_punct`    | True                                                              | 启用标点恢复         |

## 六、常见问题

- **启动报模型文件缺失**:检查第 3 节两个模型目录是否放在脚本同级目录、文件名是否一致。
- **无声音/识别不到**:确认麦克风设备,可在 `run()` 中把 `sd.InputStream` 加 `device=` 指定输入设备;终端运行 `python -m sounddevice` 查看设备列表。
- **上屏位置不对**:上屏是模拟 Ctrl+V,确保打字时光标在目标输入框内。
- **上屏后剪贴板被覆盖**:脚本会自动延时恢复原剪贴板;若原剪贴板为空或非文本(如图片)则无法恢复。
- **热词不生效**:见第 3 节说明,sherpa-onnx 1.13.x 已不支持该方式的热词文件。
