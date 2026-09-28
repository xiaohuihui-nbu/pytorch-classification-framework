# GitHub 上传、拉取与常用命令

适用于本项目，PowerShell 和 Git Bash 均可使用。除克隆命令外，以下命令都在项目根目录执行。

- 仓库：[xiaohuihui-nbu/pytorch-classification-framework](https://github.com/xiaohuihui-nbu/pytorch-classification-framework)
- 远程名：`origin`
- 主分支：`main`
- 当前远程地址：`git@github.com:xiaohuihui-nbu/pytorch-classification-framework.git`

**目录**

- [1. 日常操作速查](#1-日常操作速查)
- [2. 第一次获取项目](#2-第一次获取项目)
- [3. 上传本地修改](#3-上传本地修改)
- [4. 拉取远程更新](#4-拉取远程更新)
- [5. 分支与合并](#5-分支与合并)
- [6. 撤销与恢复](#6-撤销与恢复)
- [7. 常见问题](#7-常见问题)
- [8. 本项目的提交范围](#8-本项目的提交范围)

## 1. 日常操作速查

| 目的 | 命令 | 说明 |
|---|---|---|
| 查看本地状态 | `git status` | 判断是否有未提交文件、当前分支 |
| 查看远程地址 | `git remote -v` | 确认 push/fetch 的目标仓库 |
| 查看未暂存修改 | `git diff` | 工作区相对暂存区的差异 |
| 查看即将提交内容 | `git diff --cached` | 暂存区相对上次提交的差异 |
| 暂存全部改动 | `git add -A` | 包含新增、修改和删除；遵循忽略规则 |
| 本地提交 | `git commit -m "说明本次改动"` | 生成本地版本，不会自动上传 |
| 上传 main | `git push origin main` | 将本地 main 提交上传到 GitHub |
| 仅获取远程记录 | `git fetch origin` | 不修改当前工作区文件 |
| 更新当前 main | `git pull --ff-only origin main` | 仅允许快进更新，有分叉时停止 |
| 查看提交历史 | `git log --oneline -10` | 最近 10 次提交 |
| 查看分支跟踪关系 | `git branch -vv` | 查看当前分支及上游分支 |

`add → commit → push` 对应“选中改动 → 保存版本 → 上传版本”。尚未 commit 的内容不会通过 push 上传。

## 2. 第一次获取项目

### 2.1 克隆仓库

选择一种方式，只执行一次：

```powershell
# HTTPS：公开仓库可直接克隆
git clone https://github.com/xiaohuihui-nbu/pytorch-classification-framework.git

# 或 SSH：需要该电脑已配置 GitHub SSH 访问
git clone git@github.com:xiaohuihui-nbu/pytorch-classification-framework.git
```

进入项目并安装依赖：

```powershell
cd pytorch-classification-framework
uv sync
uv run cls --help
```

已有项目目录时，不必再次 clone。后续使用第 4 节的拉取命令更新。

### 2.2 设置提交身份

查看本仓库实际使用的身份：

```powershell
git config --get user.name
git config --get user.email
```

未配置或需要调整时，填写自己的名字及 GitHub 已验证邮箱，也可使用 GitHub 账户设置中给出的 noreply 邮箱：

```powershell
git config user.name "你的名字或GitHub用户名"
git config user.email "你的GitHub提交邮箱"
```

这里不使用 `--global`，仅设置当前仓库。提交身份与 GitHub 登录认证是两回事。

### 2.3 查看或修正远程地址

```powershell
git remote -v
```

已有 origin 但地址不正确时：

```powershell
git remote set-url origin git@github.com:xiaohuihui-nbu/pytorch-classification-framework.git
```

仅当还没有 origin 时使用：

```powershell
git remote add origin git@github.com:xiaohuihui-nbu/pytorch-classification-framework.git
```

## 3. 上传本地修改

### 3.1 在 main 上提交并上传

先执行 `git status`，确认当前分支是 main。如果正在其他分支上开发，使用第 5 节的分支流程。

```powershell
git status
git diff
git add -A
git diff --cached --stat
git diff --cached
git commit -m "完善数据处理、ImageFolder 配置与文档"
git push origin main
```

`git add -A` 会包含已删除文件。若只想上传部分文件，可以将该命令替换为：

```powershell
git add README.md docs/GIT_GUIDE.md
```

首次设置 main 的上游分支时，可使用 `git push -u origin main`。本项目已有 origin/main 跟踪关系，之后也可直接执行 `git push`。[Git push 文档](https://git-scm.com/docs/git-push)

### 3.2 确认上传结果

```powershell
git status
git fetch origin
git rev-list --left-right --count HEAD...origin/main
git log -1 --oneline
```

在 main 分支上，计数为 `0 0` 表示本地与远程 main 没有提交差异。再打开 GitHub 仓库，查看最新提交及文件。

## 4. 拉取远程更新

### 4.1 本地没有未提交修改

```powershell
git status
git switch main
git pull --ff-only origin main
uv sync
```

`--ff-only` 会在双方各有新提交、无法直接快进时停止，便于先检查差异。它不会自动替你解决分叉。[Git pull 文档](https://git-scm.com/docs/git-pull)

更新核心源码前先结束使用当前环境的训练；源码摘要和依赖版本属于严格续训契约，更新后旧 checkpoint 可能需要对应的原版本才能严格恢复。

### 4.2 本地有尚未提交的修改

可以先提交本地工作；暂时不适合提交时，使用 stash 临时保存：

```powershell
# 在当前分支保存工作区；-u 同时保存未跟踪且未被忽略的文件
git stash push -u -m "拉取前临时保存"
git pull --ff-only
git stash pop
git status
```

这里的 `git pull --ff-only` 更新**当前分支的上游**，执行前用 `git branch -vv` 确认跟踪关系。stash 不包含被忽略的 data、weights、runs、logs 等目录；pop 遇到冲突时需手工处理，不能当作自动合并成功。[Git stash 文档](https://git-scm.com/docs/git-stash)

### 4.3 本地和远程都有新提交

先提交或 stash 工作区，再获取并查看分叉：

```powershell
git fetch origin
git log --oneline --graph --decorate --all -15
```

如果当前在 main，且本地独有提交尚未推送给别人，可以将这些提交放到最新远程版本之后：

```powershell
git rebase origin/main
```

发生冲突时，打开冲突文件保留正确内容，删除冲突标记，再继续：

```powershell
git status
git add path/to/resolved_file
git rebase --continue
```

`path/to/resolved_file` 需要替换为实际文件。若要放弃本次 rebase，使用 `git rebase --abort`。完成并检查后执行 `git push origin main`；不要用强制推送跳过远程差异。若相关提交已被他人使用，应优先讨论合并方式，避免重写共享历史。

## 5. 分支与合并

### 5.1 新建开发分支

工作区干净时执行：

```powershell
git switch main
git pull --ff-only origin main
git switch -c feat/new-model

# 完成修改后
git add -A
git commit -m "增加模型配置"
git push -u origin feat/new-model
```

打开 GitHub 创建 Pull Request，以 main 为目标分支。后续在该分支继续 commit 后，执行 `git push` 即可更新同一 PR。

### 5.2 合并后更新本地

```powershell
git switch main
git pull --ff-only origin main
git branch -d feat/new-model
```

`-d` 会检查本地分支是否已合并；如果 GitHub 使用 squash 合并，可能仍拒绝删除，此时先核对提交内容，不必强制删除。

## 6. 撤销与恢复

### 6.1 取消暂存，保留文件修改

```powershell
git restore --staged README.md
```

只将 README 移出暂存区，不丢弃工作区内容。[Git restore 文档](https://git-scm.com/docs/git-restore)

### 6.2 查看或取回旧版本文件

```powershell
git log --oneline -- README.md
git show COMMIT_ID:README.md
```

先将 `COMMIT_ID` 替换为真实提交 ID，检查内容。确实需要用旧版本替换当前文件时执行：

```powershell
git restore --source COMMIT_ID -- README.md
```

这会覆盖该文件当前的工作区内容；先提交或 stash 需要保留的修改。恢复后仍需 add、commit、push 才会同步到 GitHub。

### 6.3 撤销已上传的一次普通提交

工作区干净时，用新提交反向撤销目标提交，再正常上传：

```powershell
git revert COMMIT_ID
git push origin main
```

`COMMIT_ID` 替换为需要撤销的普通提交。合并提交需要额外选择主线，不直接照搬此示例。

## 7. 常见问题

| 提示或情况 | 处理方法 |
|---|---|
| `nothing to commit` | 没有新的已暂存变化；用 git status、git diff 检查 |
| 提示未配置作者身份 | 按 2.2 节设置本仓库 user.name / user.email |
| `Permission denied (publickey)` | 执行 `ssh -T git@github.com` 检查 SSH 身份，确认该电脑公钥已加入正确账户 |
| HTTPS 方式推送要求认证 | 使用 Git 凭据管理器或个人访问令牌；不要把令牌写入仓库 URL 或文档 |
| `non-fast-forward` / `fetch first` | 远程存在本地没有的提交，按 4.3 节先获取并整合 |
| 未提交修改阻止 pull/switch | 先 commit 或 stash，避免丢失本地工作 |
| `remote origin already exists` | 使用 remote set-url，不要重复 remote add |
| `src refspec main does not match any` | 用 git branch 查看真实分支，并确认已有至少一次 commit |
| 新文件没有出现在 status 中 | 用 `git check-ignore -v 文件路径` 查看忽略来源 |
| LF/CRLF 提示 | 通常是行尾转换提示；train.sh 由 .gitattributes 固定为 LF |

## 8. 本项目的提交范围

正常提交源码、配置、脚本、文档、pyproject.toml 和 uv.lock。提交前用 `git diff --cached --stat` 检查实际范围。

以下内容按当前 `.gitignore` 留在本地：`.venv/`、data/、weights/、cache/、runs/、logs/、dist/ 和 ONNX 文件。`.gitignore` 当前还排除了 `tests/`，因此 GitHub 克隆副本不包含本地完整测试；若以后决定共享测试，再单独调整这一规则。

Git pull 不会从仓库补齐被忽略的数据和权重。新电脑先执行 `uv sync`，再按照 [README 数据准备](../README.md#3-数据准备)运行数据处理和训练命令。已有机器拉取代码不会替你重新划分数据。

模型运行方式见 [README 命令行](../README.md#5-命令行)，数据准备方式见 [README 数据准备](../README.md#3-数据准备)。完整配置 Schema 可用 `uv run cls schema --output config.schema.json` 按需生成。
