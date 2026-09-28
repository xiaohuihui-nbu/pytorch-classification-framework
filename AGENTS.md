# 项目开发约定

- 沟通和交付文档使用中文。核心包位于 src/clsframework，CLI 与 Python API 共用实现。
- 使用项目 .venv 的独立 Python 3.12，不改系统或 Conda 环境。当前 uv.lock 为 CPU 验收环境。
- 先读 README.md、docs/USAGE.md 和 docs/VALIDATION.md，区分已实测与未验收功能。
- 修改训练语义必须运行相关契约与训练测试；恢复测试要求相同环境下参数零容差对齐。
- 修改配置 schema 后重新生成 docs/config.schema.json。新能力必须拒绝不支持的配置，不得静默回退。
- 不把前向检查称为完整训练认证，不把合成数据准确率当作真实数据性能。
- 源码摘要属于严格续训契约，修改核心 Python 文件后旧 checkpoint 可能无法严格恢复，应明确说明。
- 不下载或提交大规模数据、预训练权重和运行目录到 Git；不修改相邻的 Web MVP 项目。
- 常用检查：`.venv\Scripts\python.exe -m pytest -q`，`.venv\Scripts\ruff.exe check src tests examples`，`uv build`。
