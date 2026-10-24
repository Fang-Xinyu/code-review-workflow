# CodeFlow

使用 Reviewer 和 Fixer 两个 Agent 审查、修复单个 Python 文件，通过 LangGraph 管理流程，并使用 Ruff 和 Docker 内的 pytest 检查修改结果。

设计说明见 [Design.md](Design.md)。

## 安装

需要 Python 3.11+；自动修复还需要安装并启动 Docker Desktop。

在终端进入 `codeflow` 目录后执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
```

## 启动网页

在 `codeflow` 目录执行：

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

浏览器打开终端显示的地址，通常为 http://localhost:8501。运行期间保持终端打开，按 `Ctrl+C` 停止。

## 使用

1. 在网页左侧填写 API Key、服务地址和模型名称，点击「测试模型连接」。直连 DeepSeek 时地址为 `https://api.deepseek.com`，模型名称使用账号实际支持的模型 ID。
2. 选择「仅审查」或「自动修复」，再选择演示样例，或上传、粘贴自己的 Python 代码。
3. 点击「开始任务」，完成后查看问题、修改差异和测试结果，下载代码与报告。

「仅审查」不需要 Docker。「自动修复」需先启动 Docker Desktop，再另开终端，在 `codeflow` 目录构建测试镜像并检查环境：

```bash
source .venv/bin/activate
docker build -f Dockerfile.test -t codeflow-test:1.0 .
python -m cli doctor
```

`doctor` 输出 `"status": "passed"` 后即可执行自动修复。镜像只需首次构建；后续使用保持 Docker 运行即可。演示样例自带测试，自己的代码默认由系统生成测试。

## 命令行（可选）

激活虚拟环境后，在同一终端设置模型配置。程序不会自动读取 `.env`。

```bash
export DEEPSEEK_API_KEY='你的密钥'
export DEEPSEEK_BASE_URL='https://api.deepseek.com'
export DEEPSEEK_MODEL='你的模型ID'

# 仅审查
python -m cli review --source examples/clean/solution.py

# 自动修复
python -m cli run --source examples/division/solution.py --tests examples/division/test_solution.py
```

## 测试

```bash
python -m pytest
ruff check .
```

## 注意事项

- 仅支持单个 Python 业务文件，不支持多文件项目或额外依赖安装。
- 模型调用需要有效 API Key，可能产生费用；不要提交真实密钥。
- 默认最多修复 8 轮。未完全修复时会保留部分修改及剩余问题，测试通过不代表代码完全正确。
- 网页任务记录暂存在 `runs/`，同一网页会话开始下一次任务时删除上一次记录；关网页不会立即删除，结果可手动下载。CLI 默认结束后清理，用 `--output-dir` 可保存。
