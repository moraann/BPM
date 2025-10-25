import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from threading import Thread

# 基础参数
SERIAL_PARAMS = {
    "baudrate": 115200,
    "bytesize": serial.EIGHTBITS,
    "parity": serial.PARITY_NONE,
    "stopbits": serial.STOPBITS_ONE,
    "timeout": 0.1
}
FRAME_HEADER = (0x53, 0x59)  # 帧头
FRAME_TAIL = (0x54, 0x43)    # 帧尾
RADAR_MODEL = "R60AFD1"      # 型号
PRESENCE_QUERY_CMD = bytes([0x53, 0x59, 0x80, 0x81, 0x00, 0x01, 0x0F, 0xBD, 0x54, 0x43])  # 查询命令

# 全局数据
radar_data = {
    "timestamp": [],
    "presence_state": [],  # 0=无人，1=有人
    "motion_state": [],     # 0=无目标，1=静止，2=活跃
    "motion_amplitude": [] # 0-100
}
max_data_points = 50
data_lock = False

# 校验函数
def calculate_checksum(frame_bytes):
    return sum(frame_bytes) & 0xFF

# 帧解析
def parse_radar_frame(frame_bytes):
    global radar_data, data_lock
    try:
        min_frame_len = 10  # 最小帧长（帧头2+控制字1+命令字1+长度2+数据1+校验1+帧尾2）
        if len(frame_bytes) < min_frame_len:
            return

        # 提取帧字段
        ctrl_word = frame_bytes[2]
        cmd_word = frame_bytes[3]
        len_h = frame_bytes[4]
        len_l = frame_bytes[5]
        data_len = (len_h << 8) | len_l
        data = frame_bytes[6:6+data_len]
        checksum_recv = frame_bytes[6+data_len]
        tail = (frame_bytes[7+data_len], frame_bytes[8+data_len])

        # 帧尾与校验校验
        if tail != FRAME_TAIL:
            return
        checksum_calc = calculate_checksum([FRAME_HEADER[0], FRAME_HEADER[1], ctrl_word, cmd_word, len_h, len_l, *data])
        if checksum_recv != checksum_calc:
            return

        # 数据解析
        presence = 0
        if ctrl_word == 0x80 and cmd_word == 0x01 and data_len == 1:
            presence = data[0]
        motion = 0
        if ctrl_word == 0x80 and cmd_word == 0x02 and data_len == 1:
            motion = data[0]
        amplitude = 0
        if ctrl_word == 0x80 and cmd_word == 0x03 and data_len == 1:
            amplitude = data[0]

        # 数据存储
        while data_lock:
            time.sleep(0.001)
        data_lock = True
        current_time = time.time()
        radar_data["timestamp"].append(current_time)
        radar_data["presence_state"].append(presence)
        radar_data["motion_state"].append(motion)
        radar_data["motion_amplitude"].append(amplitude)
        # 截断数据
        for key in radar_data:
            if len(radar_data[key]) > max_data_points:
                radar_data[key].pop(0)
        data_lock = False

    except Exception:
        return

# 串口读取
def serial_reader(serial_port):
    ser = None
    try:
        ser = serial.Serial(serial_port, **SERIAL_PARAMS)
        buffer = bytearray()
        cmd_send_interval = 1
        last_cmd_time = time.time()
        while True:
            # 读取数据
            if ser.in_waiting > 0:
                buffer.extend(ser.read(ser.in_waiting))
            # 发送查询命令
            if time.time() - last_cmd_time > cmd_send_interval:
                ser.write(PRESENCE_QUERY_CMD)
                last_cmd_time = time.time()
            # 解析帧
            while len(buffer) >= len(FRAME_HEADER):
                # 找帧头
                header_idx = -1
                for i in range(len(buffer)-1):
                    if buffer[i] == FRAME_HEADER[0] and buffer[i+1] == FRAME_HEADER[1]:
                        header_idx = i
                        break
                if header_idx == -1:
                    buffer.clear()
                    break
                # 判帧长
                if len(buffer) < header_idx + 6:
                    break
                len_h = buffer[header_idx+4]
                len_l = buffer[header_idx+5]
                data_len = (len_h << 8) | len_l
                total_frame_len = 2 + 1 + 1 + 2 + data_len + 1 + 2
                if len(buffer) < header_idx + total_frame_len:
                    break
                # 解析
                full_frame = buffer[header_idx:header_idx+total_frame_len]
                parse_radar_frame(full_frame)
                buffer = buffer[header_idx+total_frame_len:]
            time.sleep(0.01)
    except serial.SerialException:
        print(f"串口连接失败")
    finally:
        if ser and ser.is_open:
            ser.close()

# 可视化
def update_plot(frame, axs, line):
    global radar_data, data_lock
    # 读取数据
    while data_lock:
        time.sleep(0.001)
    data_lock = True
    current_data = {k: v.copy() for k, v in radar_data.items()}
    data_lock = False

    # 清空所有子图
    for ax in axs.flat:
        ax.clear()
        ax.axis("on")  

    # 无数据时显示提示
    if len(current_data["timestamp"]) < 1:
        for ax in axs.flat:
            ax.text(0.5, 0.5, "等待雷达数据...", ha="center", va="center", fontsize=12, color="#666")
            ax.axis("off")
        return [line]

    # 时间处理
    timestamps = [time.strftime("%H:%M:%S", time.localtime(t)) for t in current_data["timestamp"]]
    x_range = range(len(timestamps))
    tick_step = max(1, len(timestamps)//5)

    # 存在状态
    axs[0,0].set_title("人体存在状态", fontsize=10)
    latest_presence = current_data["presence_state"][-1]
    presence_text = {0:"无人", 1:"有人"}.get(latest_presence, "未知")
    color = {0:"#999", 1:"#FF6666"}.get(latest_presence, "#CCC")
    axs[0,0].text(0.5, 0.5, presence_text, ha="center", va="center", fontsize=20, color=color, fontweight="bold")
    axs[0,0].axis("off")

    # 运动状态
    axs[0,1].set_title("运动状态", fontsize=10)
    latest_motion = current_data["motion_state"][-1]
    motion_text = {0:"无目标", 1:"静止", 2:"活跃"}.get(latest_motion, "未知")
    color = {0:"#999", 1:"#FFD700", 2:"#32CD32"}.get(latest_motion, "#CCC")
    axs[0,1].text(0.5, 0.5, motion_text, ha="center", va="center", fontsize=20, color=color, fontweight="bold")
    axs[0,1].axis("off")

    # 体动幅度趋势
    axs[1,0].set_title("体动幅度趋势", fontsize=10)
    axs[1,0].plot(x_range, current_data["motion_amplitude"], 'b-', linewidth=2, label="体动幅度")
    axs[1,0].set_xlabel("时间")
    axs[1,0].set_ylabel("幅度")
    axs[1,0].set_xlim(0, max_data_points)
    axs[1,0].set_ylim(0, 100)
    axs[1,0].grid(alpha=0.3)
    axs[1,0].set_xticks(x_range[::tick_step])
    axs[1,0].set_xticklabels([timestamps[i] for i in x_range[::tick_step]], rotation=45, fontsize=8)
    axs[1,0].legend()

    # 存在状态分布
    axs[1,1].set_title(f"存在状态分布（最近{max_data_points}次）", fontsize=10)
    presence_counts = [current_data["presence_state"].count(0), current_data["presence_state"].count(1)]
    axs[1,1].pie(presence_counts, labels=["无人", "有人"], colors=["#999", "#FF6666"], autopct="%1.1f%%")

    return [line]

# -------------------------- 7. 主函数（添加启动延时）--------------------------
def main():
    # 串口选择
    available_ports = [p.device for p in serial.tools.list_ports.comports()]
    if not available_ports:
        print("无可用串口")
        return
    print("\n可用串口：")
    for i, port in enumerate(available_ports):
        print(f"  {i}: {port}")
    # 选择串口
    try:
        selected_port = available_ports[int(input(f"\n输入索引（0-{len(available_ports)-1}）："))]
    except (ValueError, IndexError):
        selected_port = available_ports[0]
        print(f"默认使用：{selected_port}")

    # 启动线程+延时
    Thread(target=serial_reader, args=(selected_port,), daemon=True).start()
    print(f"\n雷达启动中，等待2秒...")
    time.sleep(2)

    # 可视化初始化（确保axs完全创建）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # 明确创建2x2子图，避免axs为None
    fig.suptitle(f"{RADAR_MODEL}跌倒检测雷达可视化", fontsize=14)
    # 初始化线条（仅1条趋势线，避免多余线条导致错误）
    line, = axs[1,0].plot([], [], 'b-', linewidth=2)

    # 启动动画（关闭blit避免视图错误，核心修复点）
    ani = animation.FuncAnimation(
        fig, update_plot, fargs=(axs, line),
        interval=1000, blit=False,  # blit设为False，解决_get_view()错误
        save_count=max_data_points
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        import serial, matplotlib.pyplot
    except ImportError:
        print("安装依赖：pip install pyserial matplotlib")
        exit(1)
    main()