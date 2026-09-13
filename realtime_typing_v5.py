#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
稳定版流式语音输入法助手
- ASR: Sherpa-Onnx 流式 Paraformer
- ITN: cn2an 纯 Python 逆文本正则化 (免去 pynini 编译陷阱)
- 标点: CT-Transformer ONNX 离线标点恢复
- 跨平台防覆盖剪贴板上屏

- 增加监听开关
Ctrl + Shift + Space
        ↓
    开始语音输入
        ↓
     说话
        ↓
  停顿自动分句
        ↓
    自动粘贴
        ↓
Ctrl + Shift + Space
        ↓
     停止监听
"""

import os
import re
import sys
import time
import queue
import logging
import platform
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import sounddevice as sd
import sherpa_onnx
import pyperclip
import pyautogui
from pynput import keyboard

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("VoiceTyper")


@dataclass
class AppConfig:
    sample_rate: int = 16000
    block_size: int = 1600  # 100ms 帧长

    # 1. ASR 核心模型目录
    asr_model_dir: Path = Path("./sherpa-onnx-streaming-paraformer-bilingual-zh-en")
    # 下载地址：https://huggingface.co/csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en/tree/main

    # 2. 标点模型目录 (下载 punctuation-models 标签下的 release)
    punct_model_dir: Path = Path("./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12")
    # 模型下载地址： wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
    # 解压： tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
    # 下载地址： https://huggingface.co/csukuangfj/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/tree/main

    # 3. 热词列表路径
    hotwords_file: Path = Path("./hotwords.txt")

    num_threads: int = 4
    paste_delay: float = 0.05
    hotkey_str: str = "<ctrl>+<shift>+q"
    enable_itn: bool = True       # 启用 cn2an 数字转换
    enable_punct: bool = True     # 启用标点符号恢复


class TextPaster:
    """保护剪贴板的模拟输入模块"""

    def __init__(self, paste_delay: float = 0.05):
        self.paste_delay = paste_delay
        self.is_darwin = platform.system() == "Darwin"
        self.paste_keys = ("command", "v") if self.is_darwin else ("ctrl", "v")

    def paste(self, text: str):
        if not text.strip():
            return

        original_clipboard: Optional[str] = None
        try:
            original_clipboard = pyperclip.paste()
        except Exception:
            pass

        try:
            pyperclip.copy(text)
            time.sleep(self.paste_delay)
            pyautogui.hotkey(*self.paste_keys)
        finally:
            def restore():
                time.sleep(0.3)
                try:
                    if original_clipboard is not None:
                        pyperclip.copy(original_clipboard)
                except Exception:
                    pass

            threading.Thread(target=restore, daemon=True).start()


class VoiceInputEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        self.audio_queue: queue.Queue = queue.Queue(maxsize=200)
        self.is_listening = threading.Event()
        self.is_running = threading.Event()
        self.is_running.set()

        self.paster = TextPaster(paste_delay=config.paste_delay)

        # 1. 初始化 ASR
        self.recognizer = self._init_recognizer()
        self.stream = self.recognizer.create_stream()

        # 2. 初始化标点恢复器
        self.punct = self._init_punctuator() if config.enable_punct else None

        # 3. 检查 cn2an
        self.has_cn2an = False
        if self.config.enable_itn:
            try:
                import cn2an
                self.has_cn2an = True
                logger.info("已成功挂载 cn2an 逆文本正则化模块。")
            except ImportError:
                logger.warning("未安装 cn2an 库，跳过数字规范化。建议运行: pip install cn2an")

    def _init_punctuator(self) -> Optional[sherpa_onnx.OfflinePunctuation]:
        model_file = self.config.punct_model_dir / "model.onnx"
        if not model_file.is_file():
            logger.warning(f"标点模型不存在: {model_file}，将不预测标点符号。")
            return None

        logger.info("正在加载 CT-Transformer 标点模型...")
        punct_config = sherpa_onnx.OfflinePunctuationConfig(
            model=sherpa_onnx.OfflinePunctuationModelConfig(
                ct_transformer=str(model_file),
                num_threads=2,
            )
        )
        return sherpa_onnx.OfflinePunctuation(punct_config)

    def _init_recognizer(self) -> sherpa_onnx.OnlineRecognizer:
        md = self.config.asr_model_dir
        tokens = md / "tokens.txt"
        encoder = md / "encoder.int8.onnx"
        decoder = md / "decoder.int8.onnx"

        for p in [tokens, encoder, decoder]:
            if not p.is_file():
                logger.error(f"ASR 模型文件缺失: {p.resolve()}")
                sys.exit(1)

        # sherpa-onnx >= 1.13 的 from_paraformer 已移除 hotwords_file/hotwords_score
        # (新版热词改为 hr_dict_dir/hr_lexicon 同音替换方案,不再支持简单热词文件)
        if self.config.hotwords_file.is_file():
            logger.warning(
                f"当前 sherpa-onnx 版本不支持热词文件, 已忽略: {self.config.hotwords_file}"
            )

        logger.info("正在加载流式 Paraformer ASR 模型...")
        return sherpa_onnx.OnlineRecognizer.from_paraformer(
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            num_threads=self.config.num_threads,
            sample_rate=self.config.sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=300,
            debug=False,
        )

    def _post_process(self, raw_text: str) -> str:
        """核心后处理管线：ASR -> ITN(数字转阿拉伯) -> 标点预测 -> 排版优化"""
        text = raw_text.strip()
        if not text:
            return ""

        # 1. 逆文本规范化（如：六百美元 -> 600美元，三点一倍 -> 3.1倍）
        if self.has_cn2an:
            try:
                import cn2an
                text = cn2an.transform(text, "cn2an")
            except Exception as e:
                logger.warning(f"cn2an 转换异常: {e}")

        # 2. 标点符号恢复
        if self.punct:
            try:
                text = self.punct.add_punctuation(text)
            except Exception as e:
                logger.warning(f"标点恢复预测异常: {e}")

        # 3. 中英文及数字间隔美化（在中文与英文/数字之间插入合理空格）
        text = re.sub(r'([\u4e00-\u9fa5])([a-zA-Z0-9])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z0-9])([\u4e00-\u9fa5])', r'\1 \2', text)

        return text

    def _format_and_paste(self, raw_text: str):
        processed_text = self._post_process(raw_text)
        if processed_text:
            logger.info(f"📤 [上屏] -> {processed_text}")
            self.paster.paste(processed_text)

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"声卡缓冲区状态异常: {status}")

        if not self.is_listening.is_set():
            return

        try:
            self.audio_queue.put_nowait(indata.copy())
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            self.audio_queue.put_nowait(indata.copy())

    def toggle_listening(self):
        if self.is_listening.is_set():
            self.is_listening.clear()
            logger.info("🛑 已停止监听")
            self._flush_stream()
        else:
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            self.recognizer.reset(self.stream)
            self.is_listening.set()
            logger.info("🎙️ 开始语音输入（按快捷键停止）...")

    def _flush_stream(self):
        if self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        raw_text = self.recognizer.get_result(self.stream).strip()
        if raw_text:
            self._format_and_paste(raw_text)
        self.recognizer.reset(self.stream)

    def _process_audio_loop(self):
        last_text = ""
        while self.is_running.is_set():
            try:
                samples = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self.stream.accept_waveform(self.config.sample_rate, samples[:, 0])

            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)

            # 终端实时未定稿流式预览
            current_text = self.recognizer.get_result(self.stream)
            if current_text and current_text != last_text:
                sys.stdout.write(f"\r[实时识别] {current_text}")
                sys.stdout.flush()
                last_text = current_text

            # 静音断句（Endpoint）触发整句定稿
            if self.recognizer.is_endpoint(self.stream):
                raw_text = self.recognizer.get_result(self.stream).strip()
                if raw_text:
                    sys.stdout.write("\n")
                    self._format_and_paste(raw_text)

                self.recognizer.reset(self.stream)
                last_text = ""

    def run(self):
        hotkey = keyboard.HotKey(
            keyboard.HotKey.parse(self.config.hotkey_str),
            self.toggle_listening
        )

        def for_canonical(f):
            return lambda k: f(key_listener.canonical(k))

        key_listener = keyboard.Listener(
            on_press=for_canonical(hotkey.press),
            on_release=for_canonical(hotkey.release)
        )
        key_listener.daemon = True
        key_listener.start()

        worker = threading.Thread(target=self._process_audio_loop, daemon=True)
        worker.start()

        logger.info("系统初始化完成！")
        logger.info(f"👉 按快捷键 [{self.config.hotkey_str}] 切换语音输入")
        logger.info("👉 按 Ctrl + C 退出程序\n")

        try:
            with sd.InputStream(
                channels=1,
                dtype="float32",
                samplerate=self.config.sample_rate,
                blocksize=self.config.block_size,
                callback=self._audio_callback,
            ):
                while self.is_running.is_set():
                    time.sleep(0.2)
        except KeyboardInterrupt:
            logger.info("接收到终止信号，正在退出...")
        finally:
            self.is_running.clear()
            key_listener.stop()
            logger.info("资源已释放，程序退出成功。")


if __name__ == "__main__":
    config = AppConfig()
    engine = VoiceInputEngine(config)
    engine.run()