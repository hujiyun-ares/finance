# 财务 ERP 系统

## 安装步骤

### 第1步：安装 Python
到 https://www.python.org/downloads/ 下载 Python 3.10 或以上版本，安装时勾选 "Add Python to PATH"。

### 第2步：安装依赖
打开命令行（Windows 按 Win+R 输入 cmd），执行：
```
pip install -r requirements.txt
```

### 第3步：启动应用
在文件所在目录执行：
```
streamlit run financial_agent.py
```

浏览器会自动打开 http://localhost:8505

## 使用说明

- 模块一（记账）：无需 API，免费使用
- 模块二（报表）：无需 API，免费使用
- 模块三（AI 问答）：需要填写 API Key（推荐硅基流动平台）
