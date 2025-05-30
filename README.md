# Newro LLM Server

目前未实现并发，可能多请求会有bug

## 使用方法
先在 serve_models 目录下下载一个默认模型 qwen3-1.7b
```bash
git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen3-1.7B.git
```

配置conda环境，如果是windows环境下建议自行下载 cuda 版本的pytorch，然后安装 requirements.txt 中的其他库。
运行api服务：
```bash
python api.py
```
