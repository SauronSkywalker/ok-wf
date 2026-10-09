# okwf 改动说明

> 相对上游 `ok-oldking/ok-script` 的全部改动。整理于 2026-10-09。
> 环境：Windows 11 (build 26100) / Python 3.11 / AMD Ryzen 7 + Radeon 780M 核显

---

## 一、新增功能：GPU 设备选择（GUI）

### 背景

框架原本只有 `Use DirectML: Auto/Yes/No` 一个开关，只能"开不开"，不能"用哪块"。
而且 `Auto` 只查 `nvidia-smi`，**纯 AMD/Intel 机器上永远返回关**，独显/核显都用不上。

新增的 `GPU Device` 下拉框把选择权交给用户：先自动扫描本机可用显卡，再让用户挑。

### 新增文件

| 文件 | 说明 |
| --- | --- |
| `ok/util/gpu_devices.py` | DXGI 枚举、过滤/去重/优先级策略、选择解析、ONNX 会话工厂 |
| `tests/test_gpu_devices.py` | 选择策略与会话测试（35 项） |
| `tests/test_gpu_device_config.py` | GUI 接线与控件渲染测试（25 项） |

### 改动文件

| 文件 | +/- | 作用 |
| --- | --- | --- |
| `ok/util/GlobalConfig.py` | +25 | 注册 `GPU Device` 下拉框（默认 `Auto`）、扫描缓存、降级兜底 |
| `ok/__init__.py` | +92 | `og` 的 GPU 属性与 `set_gpu_device()` / `create_onnx_session()`；修复 `Auto` 判定 |
| `tests/test_web_app.py` | +2 | 补 2 处漏设的 `app.config`（修复 3 个既有失败） |
| `.gitignore` | +3 | 忽略 `work/` 临时目录 |

### 交互语义

| 下拉选项 | 行为 |
| --- | --- |
| `Auto`（默认） | 保持 `Use DirectML` 开关的原语义；选不到 GPU 自动落 CPU |
| **具体显卡名** | 明确选中 → 强制启用 DirectML 并锁定该 `device_id` |
| `CPU` | 强制关闭 DirectML，走软件推理 |

**选择实时生效**：`og.gpu_choice` / `og.gpu_device_id` 每次读配置，不缓存。
改下拉框后，下一个 `create_onnx_session()` 立即反映，**不需要重启**。

### 代码 API

```python
from ok import og

session, resolved = og.create_onnx_session("model.onnx")
print(resolved.device_id)      # 0 = DXGI 索引；None = CPU
print(resolved.resolved_label) # "AMD Radeon 780M Graphics [iGPU]"
print(session.get_providers()) # ['DmlExecutionProvider', 'CPUExecutionProvider']

# 直接用策略层
from ok.util.gpu_devices import create_session, AUTO_OPTION
session, resolved = create_session("model.onnx", AUTO_OPTION)
```

### 四条设计纪律（均有测试锁定）

1. **用 DXGI 而不是 WMI** —— DirectML 的 `device_id` 是 DXGI 索引；
   WMI 看到的是另一套设备（本机 WMI 有 GameViewer 虚拟显卡、DXGI 没有；DXGI 有 WARP、WMI 没有）。
2. **`SOFTWARE 标志` + `D3D12 能力` 两条并用** —— 微软 WARP 能创建 D3D12 设备但纯跑 CPU，
   单靠 D3D12 能力会把它当成可用 GPU。
3. **按 PCI 身份去重，LUID 不参与** —— 本机 780M 被注册成两个 DXGI 实例（LUID 69534 / 161200，
   PCI 三元组相同），不去重下拉框会出现两个同名项。去重保留首个 `device_id`。
4. **陈旧/未知选择回落 Auto** —— 显卡列表会随驱动、虚拟显示设备变化，不崩也不静默变 CPU。

### 已知限制

- 两块**型号完全相同**的独显会因 PCI 身份相同而合并成一项（已写入模块 docstring）。
- `Auto` 的核显优先级按 `dedicated VRAM < 512MB 且 shared > 1GB` 启发式判定，
  仅用于排序和 UI 标签，**不作为过滤条件**（过滤只看 SOFTWARE 标志与 D3D12 能力）。

---

## 二、缺陷修复

| # | 问题 | 根因 | 修法 |
| --- | --- | --- | --- |
| 1 | `Use DirectML: Auto` 在非 NVIDIA 机器上永远关 | 只查 `nvidia-smi`，取不到值就是 -1 → False | 抽成 `auto_dml_enabled()`；nv 值为负时回退到 DXGI 问"有没有可用显卡" |
| 2 | 3 个既有测试失败 | 测试用 `object.__new__()` 绕过 `__init__`，漏设 `app.config`；同文件另外 5 个兄弟测试都设了 | 补 `app.config = {}` 两行 |
| 3 | 运行时改下拉框，`create_onnx_session` 不生效 | `gpu_choice` 是实时读，但会话创建还依赖启动时算好的 `use_dml` | 改为：只有 `Auto` 受 `use_dml` 约束，**显式选择直接生效** |
| 4 | 测试之间配置串味，并写出 `configs/Basic Options.json` | `Config.__setitem__` 立即落盘，所有测试共用 `configs/` | 每个测试独立临时目录（`GpuConfigTestCase`）；删除被写脏的文件并回归验证 |
| 5 | `work/` 出现在 `git status` | `.git/info/exclude` 沙箱只读 | 写进 `.gitignore`（学 ok-ww 的 `working/` 做法） |
| 6 | 2 个文件报 `LF will be replaced by CRLF` | 本机 `core.autocrlf=true`，而我编辑后是 LF | 5 个涉及文件统一转 CRLF，`git diff` 仍只有 117 行真实改动（无整文件刷屏） |

### 上游问题（**本 fork 未改源码**，用安装方式绕过）

| 问题 | 现象 | 绕过方案（已验证） |
| --- | --- | --- |
| `setup.py` import 未声明的构建依赖 | `pip install -e .` 报 `ModuleNotFoundError: get_pypi_latest_version` | 两阶段安装，见下 |
| `pywin32` 被拉到 312 | 项目明确写 `pywin32>=306,!=312`，但绕开项目单独装依赖时失效 | 同上，装回项目后约束自动生效，已回到 311 |

```powershell
# 阶段1：把构建依赖装进当前环境（它们本就在 dev 组，只是构建隔离环境看不到）
.\.venv\Scripts\python.exe -m pip install --group dev
# 阶段2：无构建隔离安装，让 setup.py 用当前环境
.\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation
```

参考项目 ok-ww 的做法是 `setup.py` 用**静态 `version="0.0.1"`** 从不查 PyPI，
本 fork 保留了上游的 `dynamic version` 自动 +1 发版能力，故采用上述绕过而非改源码。

---

## 三、环境（全部在项目文件夹内，不碰 C 盘）

- **`.venv`**（gitignored）：`onnxruntime-directml 1.24.4`、PySide6 + Fluent、`opencv-python`、
  `pytest`、`fastapi/uvicorn/pywebview`、`adbutils`、`psutil`、`pywin32==311` 等
- **`ok-script 2.0.9`**：editable 安装，`ok.exe` 控制台脚本可用
- **`work/pip.ps1`**：项目内 pip 封装，把缓存设到 `work\pip-cache`（原本在 `%LOCALAPPDATA%`）
- **`.git/config` 的 `core.excludesFile`**：指向项目内 `.git\local-ignore`，
  不再读 `C:\Users\Admin\.config\git\ignore`

```powershell
.\work\pip.ps1 install <包>        # 装依赖，缓存落项目内
.\.venv\Scripts\python.exe -m ok {run_task | gui | web}
```

---

## 四、验证结果

| 项目 | 结果 |
| --- | --- |
| 全量测试 | **481 passed, 0 failed, 0 skipped**（+28 subtests，23.79s） |
| GPU 测试（venv） | **69 passed** |
| GPU 测试（裸环境，无依赖） | 68 passed + 11 优雅跳过 |
| 字节码编译 | 6 个改动文件 `py_compile` 通过 |
| CLI | `python -m ok --help` 正常 |

### 显卡推理基准（真实 YOLOv8，`echo.onnx` 36.2MB，输入 1×3×640×640）

| 后端 | mean | p95 | FPS |
| --- | --- | --- | --- |
| **DirectML（780M 核显）** | **18.9 – 21.2 ms** | 19.6 – 28.1 ms | **47 – 53** |
| CPU | 57.9 – 72.7 ms | 62.2 – 76.3 ms | 14 – 17 |

**加速 2.74x – 3.85x**

GPU 落地的三重验证：
1. `resolve(Auto)` → `device_id=0 (AMD Radeon 780M [iGPU])`，ORT 接受该 id
2. **ORT profiler 显示 `op_name='DmlFusedNode_0_0'`, `provider='DmlExecutionProvider'`** ——
   整图被 DML EP 融合执行，不是 CPU 回退
3. 实测性能显著优于 CPU

---

## 五、未处理 / 已知事项

- **上游的 `pip install -e .` 缺陷**未改源码（见上表，用两阶段安装绕过）；
  如需修根，可把 `get-pypi-latest-version` 加进 `pyproject.toml` 的 `build-system.requires`。
- **沙箱相关**：pip 默认缓存路径、全局 git ignore 都已改为项目内路径；
  若要全局生效仍需用户级配置（会写 C 盘），本 fork 未做。
- **参考仓库 `_refs/ok-wuthering-waves`**（只读副本，未修改）存在三个问题：
  `requirements.txt` 漏声明 `onnxruntime`；`OpenVinoYolo8Detect` 只试 NPU→CPU 从不试 GPU；
  OpenVINO 不支持 AMD，故在本机默认路径必然落到 CPU。你的项目应使用 `ok/util/gpu_devices.py`。

---

## 六、改动清单

```
 M .gitignore                     +3
 M ok/__init__.py                 +92
 M ok/util/GlobalConfig.py        +25
 M tests/test_web_app.py          +2
?? ok/util/gpu_devices.py              新增
?? tests/test_gpu_devices.py           新增
?? tests/test_gpu_device_config.py     新增
?? CHANGES.md                          本文档
```
