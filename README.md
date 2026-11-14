# Notes on Reinforcement Learning & Control Theory

> 本仓库已按 **Obsidian 1.15.1 Vault**（**「论文骨架 + 独立底层概念卡」** 工作流）完成整理。在 Obsidian 中通过 `Open folder as vault` 打开仓库根目录即可使用悬停预览、反向链接与知识图谱。

---

## 目录结构说明

```text
BIT/
├── 01_底层概念卡/                      # 36 张原子底层概念卡（物理直觉 + 数学形式 + 矩阵化教研实例）
│   ├── 01_控制与动力学基石/            # 输入输出定义、广义坐标、欠驱动、SRBM、全质心动力学、雅可比/微分IK、LIPM、CPG等
│   ├── 02_数值优化与混杂规划/          # 凸/非凸、OCP/MPC、混杂模态、MINLP、SQP、QP/HPIPM、MPPI/CEM、样条插值、全局图搜索等
│   ├── 03_强化学习与策略梯度/          # MDP、Bellman、TD、Actor-Critic、KL、TRPO、GAE、最大熵、DPG/TD3等
│   └── 04_腿足感知与多接触架构/        # 摩擦锥、SDF/障碍函数、WBC/HQP、ANA*/RRT、ZMP/存活性、Sim2Real、EKF状态估计、3D体素自编码器等
│
├── 1_RL/                               # 强化学习基础理论与经典算法
│   ├── Note/                           # Sutton & Barto 章节推导笔记 (Ch1–Ch13.1) + code/ + image/
│   ├── Paper/                          # 经典 RL 顶会论文笔记 (GAE, PPO, SAC, TD3)
│   └── XMIND/                          # RL 课程思维导图与导出笔记
│
├── 2_自动控制原理/                      # 经典控制理论
│   ├── Note/                           # 章节核心笔记与系统分析 (Ch1–Ch9) + image/
│   ├── PPT/                            # 课程课件与参考资料
│   └── XMIND/                          # 课程知识思维导图
│
├── 3_现代控制原理/                      # 现代控制理论
│   ├── Note/                           # 状态空间、能控能观、Lyapunov、最优控制 LQR (Ch1–Ch6)
│   └── XMIND/                          # 课程知识思维导图
│
├── 4_周报/                             # 课题每周学习汇报归档 (ignored in git)
│
├── 5_GoResearch/                       # 具身智能与多接触足式移动操作课题
│   ├── Note/                           # 前沿论文精读（含 Sleiman23 旗舰双层拆解、ReLIC、Sumo 等）+ image/
│   ├── Nvidia-Sim/                     # Isaac Sim 5.1 / Isaac Lab v2.3 / ReLIC / SUMO 复现指南
│   └── Code/                           # 本地对照开源代码仓库 (OCS2, Quad-SDK, nvblox, ignored in git)
│
├── 98_模板与归档/                      # Obsidian 标准笔记模板 + 未引用本地截图归档
│   ├── 模板/                           # 概念卡 / 论文骨架 / 周报标准模板
│   └── 未引用图片归档/                 # 清理后的冗余截图归档备份
│
└── 99_Unshared/                        # 课题调研报告与开源审查手稿 (ignored in git)
```

---

## Obsidian 使用说明

1. 打开方式：在 Obsidian 选择 `Open folder as vault` 指向本仓库根目录（内置 `.obsidian/` 已启用 `page-preview`、`backlink`、`templates`、`outline`、`graph` 与 `./image` 附件路径）。
2. 概念卡悬停与跳转：阅读笔记遇到 `[[概念卡]]` 蓝字时，**阅读视图**可直接鼠标悬停预览，**编辑/实时预览视图**按住 `Ctrl + 悬停` 触发预览（可在 Obsidian 设置 -> 核心插件 -> 页面预览 中调整）；点击进入概念卡后可按 `Alt + ←` 返回原位置，并在右侧栏查看所有反向链接论文。
