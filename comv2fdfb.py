# fd_fullbody_debug_taskspace.py —— 最初 debug 版 + 任务空间(可用 xpos 或用 FK) + 渲染修复
# 新增：实时误差绘图（Matplotlib 非阻塞）+ Viewer overlay 显示瞬时误差
# 依赖：pip install mujoco glfw matplotlib ；导出视频时需 imageio imageio-ffmpeg
from pathlib import Path
import time, sys, math
import numpy as np
import mujoco
from mujoco import mj_contactForce
# ========= 路径 =========
ROOT = Path(__file__).resolve().parent
XML_PATH  = ROOT / "RKOB_simplified_upper_with_marker_floor_down.xml"
QPOS_CSV  = ROOT / "qpos_hist.csv"
XPOS_CSV  = ROOT / "xpos_hist.csv"   # 可选：如果存在且匹配，就会被用作任务空间参考

# ========= 参考与控制参数（可修改） =========
FPS_TRAJ     = 60.0
TIMESTEP     = 5e-4
KP0          = 160.0
KD0          = 2.2 * math.sqrt(KP0)
RAMP_T       = 0.3
TORQUE_LIM   = 1500.0
SMOOTH_DERIV = True
SMOOTH_ALPHA = 0.2


# 任务空间 PD（脚/末端）
KX           = 800.0      # 位置增益（N/m）
DX           = 50.0       # 速度增益（N·s/m）

# 选择要跟踪的位置对象（优先 site，找不到退回 body）
TARGET_CANDIDATES_SITE = ("toes_r", "r_toe", "foot_r", "r_foot", "right_toe", "right_foot")
TARGET_CANDIDATES_BODY = ("calcn_r", "toes_r", "r_foot", "foot_r", "right_foot")

# ========= 新增：实时绘图与 overlay 开关 =========
ENABLE_LIVE_PLOT   = True       # True → 打开实时曲线小窗
PLOT_WINDOW_SEC    = 10.0       # 曲线显示的时间窗口长度
PLOT_UPDATE_HZ     = 30.0       # 刷新频率上限
ENABLE_OVERLAY_TXT = True       # True → 在 MuJoCo viewer 左上角显示瞬时误差数值

# ========= 工具 =========
def info(*a): print("[INFO]", *a)
def warn(*a): print("[WARN]", *a)
def err(*a):  print("[ERR ]", *a)

def load_traj(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")
    arr = np.loadtxt(csv_path, delimiter=",")
    if arr.ndim != 2:
        raise ValueError("CSV 应为二维[T, N]")
    return arr

def ramp_factor(t, T=RAMP_T):
    return 1.0 if t >= T else (t / T)

# ========= 载入 =========
assert XML_PATH.exists(), f"XML 不存在：{XML_PATH}"
qpos_ref = load_traj(QPOS_CSV)
T, nq    = qpos_ref.shape
dt_ref   = 1.0 / FPS_TRAJ
sim_duration = (T - 1) * dt_ref
info(f"qpos_ref shape = {qpos_ref.shape}, FPS={FPS_TRAJ}, dt_ref={dt_ref:.6f}s, sim_duration≈{sim_duration:.3f}s")

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data  = mujoco.MjData(model)

# ====== 新增（力矩域 PD 增益；单位与 τ 同量纲）======
KP_TAU0 = 60.0
KD_TAU0 = 2.0 * math.sqrt(KP_TAU0)   # 经验阻尼
Kp_tau  = np.full(model.nv, KP_TAU0, dtype=np.float64)
Kd_tau  = np.full(model.nv, KD_TAU0, dtype=np.float64)

if nq != model.nq:
    raise ValueError(f"CSV 列数({nq}) 与模型 nq({model.nq}) 不一致；请确认轨迹与XML匹配。")

# ========= 尝试加载 xpos_ref（可选） =========
USE_XPOS_REF = False
xpos_ref = None
if XPOS_CSV.exists():
    xr = load_traj(XPOS_CSV)  # T × (3*Nb)?
    if xr.shape[0] == T and xr.shape[1] % 3 == 0:
        nb = xr.shape[1] // 3
        try:
            xpos_ref = xr.reshape(T, nb, 3)
            if nb == model.nbody:
                USE_XPOS_REF = True
                info(f"xpos_ref loaded: T={T}, nb={nb} (matches model.nbody).")
            else:
                warn(f"xpos_ref nb={nb} 与 model.nbody={model.nbody} 不一致，改用 FK 作为任务参考。")
        except Exception as e:
            warn(f"xpos_ref 形状不匹配，改用 FK。({e})")
    else:
        warn("xpos_hist.csv 维度与预期不符，改用 FK。")

# ========= 参考速度/加速度 =========
qd_ref  = np.gradient(qpos_ref, dt_ref, axis=0)
qdd_ref = np.gradient(qd_ref,   dt_ref, axis=0)
if SMOOTH_DERIV:
    for arr in (qd_ref, qdd_ref):
        for i in range(1, T):
            arr[i] = (1 - SMOOTH_ALPHA) * arr[i-1] + SMOOTH_ALPHA * arr[i]

# ========= 初值与积分器 =========
data.qpos[:] = qpos_ref[0]
data.qvel[:] = qd_ref[0]
mujoco.mj_forward(model, data)
model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
model.opt.timestep   = TIMESTEP

# ========= 增益 =========
Kp = np.full(model.nv, KP0, dtype=np.float64)
Kd = np.full(model.nv, KD0, dtype=np.float64)

# ========= 任务空间目标对象：优先 site，其次 body =========
TARGET_IS_SITE = False
TARGET_ID = -1
for nm in TARGET_CANDIDATES_SITE:
    try:
        TARGET_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, nm)
        TARGET_IS_SITE = True
        info(f"task target: site '{nm}' (id={TARGET_ID})")
        break
    except Exception:
        pass
if TARGET_ID < 0:
    for nm in TARGET_CANDIDATES_BODY:
        try:
            TARGET_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
            TARGET_IS_SITE = False
            info(f"task target: body '{nm}' (id={TARGET_ID})")
            break
        except Exception:
            pass
if TARGET_ID < 0:
    warn("未找到合适的脚/末端对象，任务空间项将被跳过。")

# 用于 FK 的 shadow data（避免改动主 data）
data_fk = mujoco.MjData(model)

# ========= 辅助：索引映射 =========
def time_to_index(sim_t: float) -> int:
    return int(np.clip(np.floor(sim_t / dt_ref), 0, T - 1))

# ========= 新增：实时误差记录与绘图 =========
_last_plot_time = -1.0
_plot_ready = False
_t_hist = []
_eq_rms_hist = []     # qpos 误差（RMS）
_ex_hist = []         # 末端位置误差范数
_ex_comp_hist = []    # 末端位置误差分量 [ex,ey,ez]

def _init_live_plot():
    global _plot_ready, plt, fig, ax1, ax2, l_eq, l_ex, l_exx, l_exy, l_exz
    if not ENABLE_LIVE_PLOT:
        return
    try:
        import matplotlib.pyplot as plt  # 延迟导入
        plt.ion()
        fig = plt.figure(figsize=(8,6))
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2, sharex=ax1)
        ax1.set_ylabel("qpos error (RMS)")
        ax2.set_ylabel("xpos error")
        ax2.set_xlabel("time (s)")
        l_eq, = ax1.plot([], [], label="||q_d - q||")
        l_ex, = ax2.plot([], [], label="||x_d - x||")
        l_exx, = ax2.plot([], [], label="ex")
        l_exy, = ax2.plot([], [], label="ey")
        l_exz, = ax2.plot([], [], label="ez")
        ax1.legend(loc="upper right")
        ax2.legend(loc="upper right")
        _plot_ready = True
    except Exception as e:
        warn(f"实时绘图初始化失败，将仅输出 overlay 文本。({e})")
        _plot_ready = False

def _update_live_plot(sim_t: float):
    """在循环里调用，按频率限制更新曲线。"""
    global _last_plot_time
    if not (_plot_ready and ENABLE_LIVE_PLOT):
        return
    if _last_plot_time < 0 or (sim_t - _last_plot_time) >= (1.0 / PLOT_UPDATE_HZ):
        _last_plot_time = sim_t
        t0 = max(0.0, sim_t - PLOT_WINDOW_SEC)
        # 直接用全部数据，显示窗口范围
        l_eq.set_data(_t_hist, _eq_rms_hist)
        l_ex.set_data(_t_hist, _ex_hist)
        if len(_ex_comp_hist) > 0:
            ex_arr = np.array(_ex_comp_hist)
            l_exx.set_data(_t_hist, ex_arr[:,0])
            l_exy.set_data(_t_hist, ex_arr[:,1])
            l_exz.set_data(_t_hist, ex_arr[:,2])
        # 适配坐标范围
        ax1.set_xlim(t0, max(t0 + 1e-3, sim_t))
        ax2.set_xlim(t0, max(t0 + 1e-3, sim_t))
        # y 轴自适应
        if len(_eq_rms_hist) > 1:
            ax1.set_ylim(min(_eq_rms_hist)*1.05 - 1e-9, max(_eq_rms_hist)*1.05 + 1e-9)
        if len(_ex_hist) > 1:
            ax2.set_ylim(min(_ex_hist)*1.05 - 1e-9, max(_ex_hist)*1.05 + 1e-9)
        try:
            plt.pause(0.001)  # 非阻塞刷新
        except Exception:
            pass

def _append_errors(sim_t: float, e_q: np.ndarray, e_x: np.ndarray):
    """记录误差到历史（用于绘图）。"""
    _t_hist.append(sim_t)
    # qpos RMS（对 nq 归一化）
    eq_rms = float(np.linalg.norm(e_q) / math.sqrt(max(1, e_q.size)))
    _eq_rms_hist.append(eq_rms)
    # 末端位置误差
    ex_norm = float(np.linalg.norm(e_x)) if e_x is not None else 0.0
    _ex_hist.append(ex_norm)
    if e_x is None:
        _ex_comp_hist.append([0.0, 0.0, 0.0])
    else:
        _ex_comp_hist.append([float(e_x[0]), float(e_x[1]), float(e_x[2])])

# ========= 核心：一步前向动力学 + 关节PD + 任务空间PD =========
def step_once(sim_t: float):
# ---- 关节空间（改为：前馈 + 反馈力矩）----
    i = time_to_index(sim_t)
    q_d, qd_d, qdd_d = qpos_ref[i], qd_ref[i], qdd_ref[i]
    q, qd = data.qpos.copy(), data.qvel.copy()

    s = ramp_factor(sim_t)
    e, de = q_d - q, qd_d - qd


    # ① 前馈（feedforward）：用期望加速度做逆动力学 => τ_ff = M(q) qdd_d + h(q, q̇)
    
    data.qacc[:] = qdd_d                
    mujoco.mj_inverse(model, data)
    tau_ff = data.qfrc_inverse.copy()
  
    # ② 反馈（feedback）：力矩域 PD => τ_fb = Kp*e + Kd*de（加渐入）
    tau_fb = (Kp_tau * s) * e + (Kd_tau * s) * de

    # ③ 关节力矩合成
    tau = tau_ff + tau_fb
    # ---- 任务空间（末端）----
    ex_vec = None
    if TARGET_ID >= 0:
        # 当前末端位置与雅可比
        Jpos = np.zeros((3, model.nv))
        if TARGET_IS_SITE:
            x = data.site_xpos[TARGET_ID]
            mujoco.mj_jacSite(model, data, Jpos, None, TARGET_ID)
        else:
            x = data.xpos[TARGET_ID]
            mujoco.mj_jacBody(model, data, Jpos, None, TARGET_ID)
        xdot = Jpos @ data.qvel

        # 参考末端位置
        if USE_XPOS_REF and TARGET_ID < (xpos_ref.shape[1] if xpos_ref is not None else 0):
            x_d = xpos_ref[i, TARGET_ID]
            xdot_d = np.zeros(3)
        else:
            data_fk.qpos[:] = qpos_ref[i]
            data_fk.qvel[:] = 0.0
            mujoco.mj_forward(model, data_fk)
            x_d = data_fk.site_xpos[TARGET_ID] if TARGET_IS_SITE else data_fk.xpos[TARGET_ID]
            xdot_d = np.zeros(3)

        ex_vec = x_d - x
        f_task = KX * ex_vec + DX * (xdot_d - xdot)   # 末端力
        tau += Jpos.T @ f_task

    # ---- 施加力矩并步进 ----
    data.qfrc_applied[:] = np.clip(tau, -TORQUE_LIM, TORQUE_LIM)
    mujoco.mj_step(model, data)

    # ---- 记录误差用于绘图 ----
    _append_errors(sim_t, e, ex_vec)

    # 返回当前瞬时误差用于 overlay
    eq_rms = float(np.linalg.norm(e) / math.sqrt(max(1, e.size)))
    ex_norm = float(np.linalg.norm(ex_vec)) if ex_vec is not None else 0.0
    return eq_rms, ex_norm, (ex_vec if ex_vec is not None else np.zeros(3))

# ========= 优先尝试实时窗口 =========
def try_viewer():
    try:
        from mujoco import viewer
    except Exception as e:
        warn(f"加载 viewer 失败：{e}")
        return False

    info("尝试打开实时窗口（ESC 关闭）...")
    _init_live_plot()

    try:
        with viewer.launch_passive(model, data) as v:
            # 打开接触点/接触力可视化（使用 viewer 的 flags）
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

            sim_t = 0.0
            printed = False
            while sim_t < (T - 1) * dt_ref:
                eq_rms, ex_norm, ex_vec = step_once(sim_t)

                if ENABLE_OVERLAY_TXT:
                    try:
                        v.add_overlay(
                            viewer.Overlay.FastText,
                            f"qpos RMS err: {eq_rms:.4e}",
                            f"xpos err |norm|: {ex_norm:.4e}   ex=({ex_vec[0]:+.3e},{ex_vec[1]:+.3e},{ex_vec[2]:+.3e})"
                        )
                    except Exception:
                        try:
                            v.add_overlay(viewer.GRID_TOPLEFT, "qpos RMS err", f"{eq_rms:.4e}")
                            v.add_overlay(viewer.GRID_TOPLEFT, "xpos |err|", f"{ex_norm:.4e}")
                        except Exception:
                            pass

                _update_live_plot(sim_t)

                v.sync()
                sim_t += model.opt.timestep

                if not printed and sim_t > 2.0:
                    info("仿真进行中... (已过2s)")
                    printed = True

        info("仿真结束")

            
        return True
    except Exception as e:
        warn(f"viewer 运行失败：{e}")
        return False

# =========  离线渲染到 mp4（修复离屏缓冲大小） =========
def render_to_mp4(path="fd_debug.mp4", seconds=None, width=960, height=540, fps=60):
    # 尝试导出需要 imageio
    try:
        import imageio
    except ImportError:
        err("未安装 imageio，无法导出视频。请先安装: pip install imageio imageio-ffmpeg")
        return

    # 确保 offscreen framebuffer 足够大
    vis_gl = getattr(model.vis, "global_")
    if getattr(vis_gl, "offwidth", 0)  < width:  vis_gl.offwidth  = int(width)
    if hasattr(vis_gl, "offheight") and getattr(vis_gl, "offheight", 0) < height: 
        vis_gl.offheight = int(height)
    if getattr(vis_gl, "offwidth", 0)  < width:  vis_gl.offwidth  = int(width)
    if hasattr(vis_gl, "offheight") and getattr(vis_gl, "offheight", 0) < height: vis_gl.offheight = int(height)

    # seconds 缺省则导出整段
    total_sim = (T - 1) * dt_ref
    if seconds is None:
        seconds = total_sim

    renderer = mujoco.Renderer(model, width=width, height=height) 
    n_frames = int(round(seconds * fps))
    frame_dt = 1.0 / fps
    info(f"离线渲染 {seconds:.3f}s → {path} (fps={fps})")
    sim_t = 0.0

    # 为了导出后可看误差趋势，顺便在末尾保存静态 PNG
    for_plot_png = "errors.png"

    with imageio.get_writer(path, fps=fps) as w:
        for _ in range(n_frames):
            target = min(sim_t + frame_dt, total_sim)
            while sim_t < target:
                eq_rms, ex_norm, _ = step_once(sim_t)
                sim_t += model.opt.timestep
            renderer.update_scene(data)
            w.append_data(renderer.render())
            if sim_t >= total_sim:
                break
    info("导出完成")

    # 保存静态误差曲线
    try:
        import matplotlib.pyplot as plt
        t = np.array(_t_hist)
        eq = np.array(_eq_rms_hist)
        ex = np.array(_ex_hist)
        plt.figure(figsize=(8,4))
        plt.plot(t, eq, label="qpos RMS error")
        plt.plot(t, ex, label="xpos |error|")
        plt.xlabel("time (s)")
        plt.ylabel("error")
        plt.legend()
        plt.tight_layout()
        plt.savefig(for_plot_png, dpi=150)
        info(f"误差曲线已保存为 {for_plot_png}")
    except Exception as e:
        warn(f"保存误差曲线失败：{e}")

# ========= 运行 =========
if not try_viewer():
    try:
        render_to_mp4()
    except Exception as e:
        err(f"离线渲染也失败：{e}")
        sys.exit(1)
