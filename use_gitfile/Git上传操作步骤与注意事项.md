# Git 仓库上传操作步骤与注意事项

本文档适用于本仓库：

```text
https://github.com/LYJ101415/trt_study.git
```

主要内容：

1. 日常上传的标准流程
2. `non-fast-forward` 推送失败的处理
3. 未跟踪文件与远端文件冲突的处理
4. 大文件、`.gitignore`、分支保护、认证
5. 冲突处理与常用回滚命令
6. 本仓库这次实际处理的记录

---

## 1. 上传前的三分钟检查

进入仓库目录：

```powershell
cd D:\CV_FILE\trt_study
```

先看状态：

```powershell
git status -sb
git remote -v
git log --oneline --graph --decorate -12 --all
```

重点看 `git status -sb` 的第一行：

```text
## main...origin/main [ahead 1, behind 8]
```

含义：

| 标记 | 含义 |
|---|---|
| `ahead 1` | 本地比远端多 1 个提交，需要 push |
| `behind 8` | 远端比本地多 8 个提交，push 前必须先同步 |
| `ahead 1, behind 8` | 本地和远端已经分叉，不能直接 push |
| 没有 ahead/behind | 本地和远端一致 |

> `git status` 里的 ahead/behind 基于最近一次 `git fetch`。
> 推送前建议先执行一次 `git fetch origin`，避免看到过期状态。

---

## 2. 日常上传标准流程

### 2.1 查看改了什么

```powershell
git status
git diff
git diff --staged
```

### 2.2 只添加需要上传的文件

```powershell
git add path\to\file1
git add path\to\dir
```

不建议无脑使用：

```powershell
git add .
```

原因：

- 容易把编译产物、模型、数据集、临时文件一起提交；
- 大文件一旦进入历史，很难彻底删除；
- 可能把本地私密配置（token、路径、账号）推到公开仓库。

添加后确认：

```powershell
git status
git diff --staged --stat
```

### 2.3 提交

```powershell
git commit -m "feat: 添加 xxx 功能"
```

常用提交前缀：

```text
feat:     新功能
fix:      修复问题
docs:     文档
refactor: 重构
perf:     性能优化
build:    构建脚本/依赖
chore:    杂项
```

### 2.4 同步远端

```powershell
git fetch origin
git status -sb
```

如果本地没有落后：

```powershell
git push origin main
```

如果本地落后，或显示 `behind`：

```powershell
git rebase origin/main
git push origin main
```

如果本地没有任何提交，只是同步：

```powershell
git pull --rebase origin main
```

### 2.5 推送后确认

```powershell
git status -sb
git log --oneline -3
```

正常结果：

```text
## main...origin/main
```

没有 `ahead` / `behind` 即表示同步完成。

---

## 3. `non-fast-forward` 报错怎么处理

典型报错：

```text
! [rejected]        main -> main (non-fast-forward)
error: failed to push some refs to 'https://github.com/...'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart.
```

### 3.1 原因

远端 `main` 上有你本地没有的提交。Git 拒绝推送，是为了防止你覆盖别人的提交。

处理原则：

```text
先同步远端 → 解决冲突 → 再推送
```

### 3.2 方案 A：rebase（推荐，历史更干净）

```powershell
git fetch origin
git rebase origin/main
```

如果成功：

```powershell
git push origin main
```

rebase 会把你的本地提交“重放”到远端最新提交之后，提交哈希会变化，这是正常现象。

### 3.3 方案 B：merge（保留分叉历史）

```powershell
git fetch origin
git merge origin/main
git push origin main
```

会多一个合并提交，适合团队约定使用 merge 的情况。

### 3.4 方案 C：force push（高风险，不推荐）

```powershell
git push --force-with-lease origin main
```

只有当**确认远端新增提交全部不要**时才使用。

风险：

- 覆盖远端提交；
- 其他人基于这些提交的工作会丢失；
- 分支保护开启时可能直接失败。

本仓库这次的情况**不应该**使用 force push，因为远端有 8 个正常提交。

---

## 4. 未跟踪文件冲突：pull/rebase 前必须先处理

### 4.1 报错样子

```text
error: The following untracked working tree files would be overwritten by checkout:
        cpp_pipeline/make_cpp/main.cpp
Please move or remove them before you merge.
```

### 4.2 原因

本地有未跟踪文件（`??`），远端新增了同路径文件。Git 不敢覆盖你本地未提交的内容，所以中止同步。

### 4.3 如何提前发现

列出本地未跟踪文件：

```powershell
git status --porcelain -uall
```

列出远端相对本地新增/修改的文件：

```powershell
git diff --name-only $(git merge-base HEAD origin/main)..origin/main
```

PowerShell 里如果 `$()` 不生效，可以分两步：

```powershell
$base = git merge-base HEAD origin/main
git diff --name-only "$base..origin/main"
```

找交集：两边都出现的路径，就是 pull/rebase 时会冲突的文件。

### 4.4 处理原则

1. **先备份**，不要直接删除；
2. 把冲突文件移出仓库，或改成远端没有的路径；
3. 执行 `git pull --rebase` / `git rebase`；
4. 同步完成后，再对比备份与远端版本，决定保留哪一份。

### 4.5 备份并移走冲突目录（PowerShell）

```powershell
cd D:\CV_FILE\trt_study

$bk = "D:\CV_FILE\trt_study_untracked_backup"
New-Item -ItemType Directory -Force $bk | Out-Null

Move-Item .\cpp_pipeline\make_cpp      "$bk\make_cpp"
Move-Item .\cpp_pipeline\my_check_env  "$bk\my_check_env"
```

然后：

```powershell
git fetch origin
git rebase origin/main
```

### 4.6 判断本地文件和远端是否相同

```powershell
git hash-object .\cpp_pipeline\make_cpp\main.cpp
git rev-parse origin/main:cpp_pipeline/make_cpp/main.cpp
```

两个哈希相同 → 内容一致，本地这份可以直接丢弃；
不同 → 需要人工对比后再决定。

---

## 5. 大文件与编译产物

### 5.1 GitHub 限制

- 单文件超过 **50 MB**：警告；
- 单文件超过 **100 MB**：直接拒绝推送；
- 仓库长期包含大量二进制文件，会让 clone 越来越慢。

### 5.2 不应该提交的内容

```text
build/  out/  cmake-build-*/
*.o  *.obj  *.so  *.dll  *.a  *.lib
*.exe  *.bin
*.engine  *.onnx  *.pt  *.pth  *.trt
数据集图片、视频、日志
__pycache__/  .venv/  venv/
.idea/  .vscode/  *.user
```

### 5.3 `.gitignore` 示例

在仓库根目录创建或修改 `.gitignore`：

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/

# C/C++
build/
out/
cmake-build-*/
*.o
*.obj
*.so
*.dll
*.a
*.lib
*.exe

# TensorRT / 模型
*.engine
*.onnx
*.pt
*.pth
*.trt

# IDE
.idea/
.vscode/
*.user

# 日志与临时文件
*.log
tmp/
```

注意：`.gitignore` 只对**尚未跟踪**的文件生效。已经提交过的大文件不会因为加了 `.gitignore` 就消失，需要从历史中删除。

### 5.4 已经提交了大文件怎么办

如果只是最近一次提交且还没推送：

```powershell
git rm --cached path\to\bigfile
git commit --amend
```

如果已经推送：

- 简单情况：`git rm --cached` 后提交，文件仍留在历史里；
- 彻底清理历史：使用 `git filter-repo` 或 BFG，属于高风险操作，必须先备份仓库。

本仓库历史上已经出现过“从仓库移除大文件，仅保留本地”的提交，说明大文件问题已经发生过一次。后续提交前务必检查：

```powershell
git diff --staged --stat
```

---

## 6. 分支保护与 Pull Request

如果推送时看到：

```text
remote: - Changes must be made through a pull request.
```

说明仓库开启了分支保护，正常流程应该是：

```powershell
git switch -c feature/my-change
git add ...
git commit -m "feat: ..."
git push -u origin feature/my-change
```

然后在 GitHub 上创建 Pull Request，审核通过后合并到 `main`。

如果账号有管理员 bypass 权限，Git 可能提示：

```text
remote: Bypassed rule violations for refs/heads/main
```

这表示规则被跳过，推送仍然成功。不要把这个提示当成正常流程；团队协作时应走 PR。

---

## 7. 认证方式

### 7.1 HTTPS + Personal Access Token

GitHub 已不支持账号密码推送。使用 HTTPS 时，需要 Personal Access Token（PAT）。

不要把 token 写进仓库文件，也不要把带 token 的 URL 提交到任何地方。

查看当前远端：

```powershell
git remote -v
```

修改远端地址：

```powershell
git remote set-url origin https://github.com/LYJ101415/trt_study.git
```

### 7.2 SSH

如果本机已经有 SSH key，可以改用 SSH：

```powershell
git remote set-url origin git@github.com:LYJ101415/trt_study.git
ssh -T git@github.com
```

`~/.ssh/config` 中可以配置多个 GitHub 账号对应的 Host，例如：

```text
Host github-lyj
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_lyj
```

对应远端：

```powershell
git remote set-url origin git@github-lyj:LYJ101415/trt_study.git
```

---

## 8. 冲突处理

### 8.1 rebase 冲突

```powershell
git status
```

编辑冲突文件，删除 `<<<<<<<`、`=======`、`>>>>>>>` 标记后：

```powershell
git add 冲突文件
git rebase --continue
```

放弃本次 rebase：

```powershell
git rebase --abort
```

### 8.2 merge 冲突

```powershell
git status
# 编辑冲突文件
git add 冲突文件
git commit
```

放弃本次 merge：

```powershell
git merge --abort
```

---

## 9. 常用撤销与恢复

### 9.1 取消暂存

```powershell
git restore --staged path\to\file
```

### 9.2 丢弃工作区修改（危险）

```powershell
git restore path\to\file
```

### 9.3 修改最后一次提交信息

```powershell
git commit --amend -m "新的提交信息"
```

### 9.4 回退提交

```powershell
# 保留工作区修改，取消提交
git reset --soft HEAD~1

# 取消提交并取消暂存，保留文件修改
git reset HEAD~1

# 彻底丢弃提交和修改（危险）
git reset --hard HEAD~1
```

已经推送到远端的提交，不建议 `reset --hard` 后强推，优先使用：

```powershell
git revert <commit>
```

### 9.5 找回误删的提交/分支

```powershell
git reflog
git switch -c recover-branch <commit-id>
```

---

## 10. 本仓库这次的实际处理记录

### 10.1 处理前状态

```text
## main...origin/main [ahead 1, behind 8]
```

- 本地提交：`50329df 添加项目代码`
- 远端最新：`00aa13d 从仓库移除大文件，仅保留本地`
- 本地提交只新增 `all_gpu_asycn_noNEDEC_3/`
- 远端 8 个提交没有修改该目录

### 10.2 冲突预判

本地未跟踪文件与远端新增文件有 23 个路径重叠：

```text
cpp_pipeline/make_cpp/...
cpp_pipeline/my_check_env/...
```

其中 22 个内容完全相同，1 个不同：

```text
cpp_pipeline/make_cpp/my_readme.md
```

本地是旧版，远端是新版。

### 10.3 实际执行

1. 备份并移走冲突目录：

```text
D:\CV_FILE\trt_study_untracked_backup_20261102\
    make_cpp\
    my_check_env\
```

2. 同步远端：

```powershell
git fetch origin
git rebase origin/main
```

3. rebase 成功，新提交：

```text
597ad36 添加项目代码
```

4. 推送：

```powershell
git push origin main
```

结果：

```text
00aa13d..597ad36  main -> main
```

5. 推送后状态：

```text
## main...origin/main
```

### 10.4 备注

推送时 GitHub 返回：

```text
remote: Bypassed rule violations for refs/heads/main:
remote: - Changes must be made through a pull request.
```

说明仓库开启了 PR 规则，但当前账号有 bypass 权限。团队协作时应优先走分支 + PR。

---

## 11. 上传前检查清单

每次 push 前逐项确认：

- [ ] `git status -sb` 看清 ahead / behind
- [ ] `git diff --staged --stat` 确认没有误加文件
- [ ] 没有提交 `.engine`、`.onnx`、`.pt`、数据集、视频、日志
- [ ] 没有提交 token、密码、私钥、内网地址
- [ ] 单个文件不超过 100 MB
- [ ] 先 `git fetch origin`，再 `git rebase origin/main`
- [ ] rebase/merge 冲突已解决
- [ ] `git log --oneline -5` 提交信息清晰
- [ ] push 后 `git status -sb` 没有 ahead/behind
- [ ] 团队仓库走分支 + PR，不直接推 `main`

---

## 12. 常用命令速查

```powershell
# 状态
git status -sb
git log --oneline --graph --decorate -12 --all
git remote -v

# 暂存与提交
git add path\to\file
git diff --staged --stat
git commit -m "feat: ..."

# 同步
git fetch origin
git rebase origin/main

# 推送
git push origin main

# 分支
git switch -c feature/xxx
git switch main
git branch -a

# 冲突
git status
git add 冲突文件
git rebase --continue
git rebase --abort

# 撤销
git restore --staged path\to\file
git restore path\to\file
git reset --soft HEAD~1
git revert <commit>
git reflog
```

---

## 13. 一句话原则

```text
先看状态，再提交；
先同步，再推送；
大文件不进库，token 不进库；
不确定就备份，不要 force push。
```
