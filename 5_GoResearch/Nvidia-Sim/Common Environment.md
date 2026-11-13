# 1. 修复容器 DNS 解析异常与并发丢包（K8s 特有问题）

- **故障现象**：执行 `conda` 或 `socket.gethostbyname` 时报错 `[Errno -3] Temporary failure in name resolution`。

- **核心原因**：容器 `/etc/resolv.conf` 默认配置了 `options ndots:5`，导致解析普通域名时无谓追加集群内部域名（如 `.kube.local`）；同时 Linux 内核进行 NAT 转发（Conntrack）时，对 glibc 并发的 IPv4/IPv6 查询存在竞态丢包 Bug。

- **有效解决方案**： 直接覆盖 `/etc/resolv.conf`，将公网 DNS 置顶，限制 `ndots` 为 1，并开启分离端口重试：

  Bash

  ```bash
  cat << 'EOF' > /etc/resolv.conf
  nameserver 223.5.5.5
  nameserver 114.114.114.114
  nameserver 10.200.0.2
  options ndots:1 timeout:2 attempts:2 single-request-reopen
  EOF
  ```

# 2. 解决 Conda 官方服务条款阻断与换国内源

- **故障现象**：报 `CondaToSNonInteractiveError: Terms of Service have not been accepted`，且官方源下载极慢。

- **核心原因**：Anaconda 官方对商业频道增加了强制协议验证，结合海外链路不通直接阻断。

- **有效解决方案**： 将 Conda 完全切换至国内镜像源（避开官方 ToS 机制），清除缓存后创建环境：

  Bash

  ```bash
  # 1. 写入清华镜像源配置
  cat << 'EOF' > ~/.condarc
  channels:
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  show_channel_urls: true
  default_channels:
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  custom_channels:
    conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/
  EOF
  
  # 2. 清空索引缓存
  conda clean -i -y
  
  # 3. 创建目标环境
  conda create -n GS python=3.10 -y
  ```

# 3. 彻底清除 Pip 中失效的 NGC 源与重试卡顿

- **故障现象**：每次使用 `pip` 安装时，均出现 `Failed to resolve 'pypi.ngc.nvidia.com'` 并强制重试 5 次，卡顿约 1 分钟。

- **核心原因**：NVIDIA 容器镜像在系统多处全局路径（如 `/etc/pip.conf`、`/etc/xdg/pip/pip.conf` 等）注入了 NGC 依赖源，仅用 `pip config unset` 只能删除当前用户目录配置，无法清除系统级配置。

- **有效解决方案**： 批量删除所有分散配置文件，统一将系统全局源设置为阿里云镜像：

  Bash

  ```bash
  # 1. 彻底移除各级残留配置文件
  rm -f /etc/xdg/pip/pip.conf \
        /etc/pip.conf \
        /root/.pip/pip.conf \
        /root/.config/pip/pip.conf \
        /root/miniconda3/envs/GS/pip.conf \
        /root/miniconda3/pip.conf
  
  # 2. 写入干净的全局源（不包含任何 extra-index-url）
  cat << 'EOF' > /etc/pip.conf
  [global]
  index-url = https://mirrors.aliyun.com/pypi/simple/
  trusted-host = mirrors.aliyun.com
  timeout = 60
  EOF
  ```

