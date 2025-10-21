import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import struct
import time
from threading import Thread

# 串口参数
SERIAL_PARAMS = {
    "baudrate": 9600,
    "bytesize": serial.EIGHTBITS,
    "parity": serial.PARITY_NONE,
    "stopbits": serial.STOPBITS_ONE,
    "timeout": 0.1
}
FRAME_START = 0x55  # 帧起始码
# CRC16校验表
CRC_HI = [
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
    0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
    0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41
]

CRC_LO = [
    0x00, 0xC0, 0xC1, 0x01, 0xC3, 0x03, 0x02, 0xC2, 0xC6, 0x06, 0x07, 0xC7,
    0x05, 0xC5, 0xC4, 0x04, 0xCC, 0x0C, 0x0D, 0xCD, 0x0F, 0xCF, 0xCE, 0x0E,
    0x0A, 0xCA, 0xCB, 0x0B, 0xC9, 0x09, 0x08, 0xC8, 0xD8, 0x18, 0x19, 0xD9,
    0x1B, 0xDB, 0xDA, 0x1A, 0x1E, 0xDE, 0xDF, 0x1F, 0xDD, 0x1D, 0x1C, 0xDC,
    0x14, 0xD4, 0xD5, 0x15, 0xD7, 0x17, 0x16, 0xD6, 0xD2, 0x12, 0x13, 0xD3,
    0x11, 0xD1, 0xD0, 0x10, 0xF0, 0x30, 0x31, 0xF1, 0x33, 0xF3, 0xF2, 0x32,
    0x36, 0xF6, 0xF7, 0x37, 0xF5, 0x35, 0x34, 0xF4, 0x3C, 0xFC, 0xFD, 0x3D,
    0xFF, 0x3F, 0x3E, 0xFE, 0xFA, 0x3A, 0x3B, 0xFB, 0x39, 0xF9, 0xF8, 0x38,
    0x28, 0xE8, 0xE9, 0x29, 0xEB, 0x2B, 0x2A, 0xEA, 0xEE, 0x2E, 0x2F, 0xEF,
    0x2D, 0xED, 0xEC, 0x2C, 0xE4, 0x24, 0x25, 0xE5, 0x27, 0xE7, 0xE6, 0x26,
    0x22, 0xE2, 0xE3, 0x23, 0xE1, 0x21, 0x20, 0xE0, 0xA0, 0x60, 0x61, 0xA1,
    0x63, 0xA3, 0xA2, 0x62, 0x66, 0xA6, 0xA7, 0x67, 0xA5, 0x65, 0x64, 0xA4,
    0x6C, 0xAC, 0xAD, 0x6D, 0xAF, 0x6F, 0x6E, 0xAE, 0xAA, 0x6A, 0x6B, 0xAB,
    0x69, 0xA9, 0xA8, 0x68, 0x78, 0xB8, 0xB9, 0x79, 0xBB, 0x7B, 0x7A, 0xBA,
    0xBE, 0x7E, 0x7F, 0xBF, 0x7D, 0xBD, 0xBC, 0x7C, 0xB4, 0x74, 0x75, 0xB5,
    0x77, 0xB7, 0xB6, 0x76, 0x72, 0xB2, 0xB3, 0x73, 0xB1, 0x71, 0x70, 0xB0
]

# 全局数据存储
radar_data = {
    "timestamp": [],       # 时间戳
    "presence_state": [],  # 0=无人，1=有人静止，2=有人运动
    "motion_amplitude": [],# 0-100
    "vital_sign": []       # 4字节Float
}
init_success = False  # 初始化成功标志
max_data_points = 50  
data_lock = False    


# CRC16校验
def calculate_crc16(data_bytes):
    crc_hi = 0xFF
    crc_lo = 0xFF
    for byte in data_bytes:
        index = (crc_lo ^ byte) & 0xFF 
        crc_lo = crc_hi ^ CRC_HI[index]
        crc_hi = CRC_LO[index]
    return (crc_lo, crc_hi)


# 帧解析
def parse_radar_frame(frame_bytes):
    global radar_data, data_lock, init_success
    try:
        # 最小帧长8Byte
        if len(frame_bytes) < 8:
            print(f"帧长不足（{len(frame_bytes)}Byte），跳过")
            return

        # 提取帧字段
        lenth_l = frame_bytes[1]
        lenth_h = frame_bytes[2]
        func_code = frame_bytes[3]
        addr1 = frame_bytes[4]
        addr2 = frame_bytes[5]
        data_segment = frame_bytes[6:-2]
        crc_recv_lo = frame_bytes[-2]
        crc_recv_hi = frame_bytes[-1]

        # CRC校验
        crc_calc_data = frame_bytes[1:-2]
        crc_calc_lo, crc_calc_hi = calculate_crc16(crc_calc_data)
        if crc_recv_lo != crc_calc_lo or crc_recv_hi != crc_calc_hi:
            print(f"CRC校验失败：接收(0x{crc_recv_lo:02X},0x{crc_recv_hi:02X})，计算(0x{crc_calc_lo:02X},0x{crc_calc_hi:02X})")
            return
        
        '''
        # 功能码0X03（被动上报）+ 地址码1=0X05（其他信息）对应初始化相关指令
        if func_code == 0x03 and addr1 == 0x05:
            # 1. 异常复位指令
            if len(data_segment) == 2 and data_segment == [0x02, 0x0F]:
                print(f"[{time.strftime('%H:%M:%S')}] 雷达上电：收到异常复位指令（0X02 0X0F），开始初始化...")
            # 2. 初始化成功指令
            elif len(data_segment) == 2 and data_segment == [0x0A, 0x0F]:
                global init_success
                init_success = True
                print(f"[{time.strftime('%H:%M:%S')}] 雷达初始化成功：收到初始化成功指令（0X0A 0X0F），开始接收业务数据...")
            return  # 初始化指令无需后续业务数据解析

        # 仅初始化成功后，才解析业务数据（环境状态、体征参数）
        if not init_success:
            print(f"[{time.strftime('%H:%M:%S')}] 等待雷达初始化...（未收到初始化成功指令）")
            return
        '''
        init_success = True  # 直接设为已初始化，简化测试

        # 业务数据解析
        # 1. 解析存在状态（手册9.2章：addr1=0x03，addr2=0x05）
        presence_state = 0
        if func_code in [0x03, 0x04] and addr1 == 0x03 and addr2 == 0x05:
            if len(data_segment) == 3:
                if data_segment == [0x00, 0xFF, 0xFF]:
                    presence_state = 0
                elif data_segment == [0x01, 0x00, 0xFF]:
                    presence_state = 1
                elif data_segment == [0x01, 0x01, 0x01]:
                    presence_state = 2
            else:
                print(f"环境状态数据段错（{len(data_segment)}Byte），需3Byte（手册9.2章）")
                return

        # 2. 解析体动幅度+体征参数（手册9.2章：addr2=0x06）
        motion_amplitude = 0
        vital_sign = 0.0
        if func_code in [0x03, 0x04] and addr1 == 0x03 and addr2 == 0x06:
            if len(data_segment) == 4:
                vital_sign = struct.unpack("<f", bytes(data_segment))[0]
                motion_amplitude = max(0, min(100, int(round(vital_sign * 10))))
            else:
                print(f"体征参数数据段错（{len(data_segment)}Byte），需4Byte（手册9.2章）")
                return

        # 存储数据
        while data_lock:
            time.sleep(0.001)
        data_lock = True
        current_time = time.time()
        radar_data["timestamp"].append(current_time)
        radar_data["presence_state"].append(presence_state)
        radar_data["motion_amplitude"].append(motion_amplitude)
        radar_data["vital_sign"].append(vital_sign)

        # 截断数据
        for key in radar_data:
            if len(radar_data[key]) > max_data_points:
                radar_data[key].pop(0)
        data_lock = False

        # 打印业务数据
        state_map = {0: "无人", 1: "有人静止", 2: "有人运动"}
        print(f"[{time.strftime('%H:%M:%S')}] 状态：{state_map[presence_state]} | 体动幅度：{motion_amplitude} | 体征：{vital_sign:.2f}")

    except Exception as e:
        frame_str = [hex(b) for b in frame_bytes[:10]]
        print(f"帧解析错：{e}，帧前10Byte：{frame_str}...")


# 串口读取
def serial_reader(serial_port):
    ser = None
    try:
        ser = serial.Serial(serial_port, **SERIAL_PARAMS)
        print(f"已连R24AVD1雷达：{serial_port}")

        # 环境状态查询命令
        read_cmd = [0x55, 0x07, 0x00, 0x01, 0x04, 0x0C, 0xEA, 0xDB]
        ser.write(bytes(read_cmd))
        print("已发环境状态查询命令")

        buffer = bytearray()
        while True:
            if ser.in_waiting > 0:
                buffer.extend(ser.read(ser.in_waiting))
                print(f"当前缓存数据：{[hex(b) for b in buffer]}")  # 新增打印，确认是否有数据
                ser.write(bytes(read_cmd))
                print("已发环境状态查询命令")
                while len(buffer) >= 3:
                    start_idx = buffer.find(FRAME_START)
                    if start_idx == -1:
                        buffer.clear()
                        break

                    # 计算总帧长（手册9.1章）
                    if len(buffer) < start_idx + 3:
                        break
                    data_len = (buffer[start_idx + 2] << 8) | buffer[start_idx + 1]
                    total_frame_len = 1 + 2 + data_len
                    if total_frame_len < 8:
                        print(f"非法帧长（{total_frame_len}Byte），跳过（手册≥8Byte）")
                        buffer = buffer[start_idx + 1:]
                        continue
                    if len(buffer) < start_idx + total_frame_len:
                        break

                    # 解析帧
                    frame = buffer[start_idx:start_idx + total_frame_len]
                    parse_radar_frame(frame)
                    buffer = buffer[start_idx + total_frame_len:]

            time.sleep(0.01)

    except serial.SerialException as e:
        print(f"串口错：{e}（检查接线，手册3.2章：雷达TX→USB RX，雷达RX→USB TX）")
    except Exception as e:
        print(f"读取线程错：{e}")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("串口已关")


# -------------------------- 可视化（不变）--------------------------
def update_plot(frame, axs, lines):
    global radar_data, data_lock, init_success
    while data_lock:
        time.sleep(0.001)
    data_lock = True
    current_data = {k: v.copy() for k, v in radar_data.items()}
    data_lock = False

    # 未初始化时，显示初始化状态
    if not init_success:
        axs[0, 0].clear()
        axs[0, 0].text(0.5, 0.5, "雷达初始化中...\n（等待0X0A 0X0F指令）", ha="center", va="center", fontsize=16, color="#666666")
        axs[0, 0].axis("off")
        return lines

    # 有数据时更新图表
    if len(current_data["timestamp"]) < 2:
        return lines

    timestamps = [time.strftime("%H:%M:%S", time.localtime(t)) for t in current_data["timestamp"]]
    x_range = range(len(timestamps))
    tick_step = max(1, len(timestamps) // 5)

    # 子图1：存在状态
    axs[0, 0].clear()
    axs[0, 0].set_title("人体存在状态（手册7.1章DP1）", fontsize=11)
    latest_state = current_data["presence_state"][-1]
    state_text = {0: "无人", 1: "有人静止", 2: "有人运动"}[latest_state]
    color = {0: "#999999", 1: "#FFD700", 2: "#FF6666"}[latest_state]
    axs[0, 0].text(0.5, 0.5, state_text, ha="center", va="center", fontsize=22, color=color, fontweight="bold")
    axs[0, 0].axis("off")

    # 子图2：体动幅度
    lines[0].set_data(x_range, current_data["motion_amplitude"])
    axs[0, 1].set_title("体动幅度（0-100，手册7.1章DP4）", fontsize=11)
    axs[0, 1].set_xlabel("时间")
    axs[0, 1].set_ylabel("幅度")
    axs[0, 1].set_xlim(0, max_data_points)
    axs[0, 1].set_ylim(0, 100)
    axs[0, 1].grid(alpha=0.3)
    axs[0, 1].set_xticks(x_range[::tick_step])
    axs[0, 1].set_xticklabels([timestamps[i] for i in x_range[::tick_step]], rotation=45, fontsize=8)

    # 子图3：体征参数
    lines[1].set_data(x_range, current_data["vital_sign"])
    axs[1, 0].set_title("体征参数（Float，手册9.2章）", fontsize=11)
    axs[1, 0].set_xlabel("时间")
    axs[1, 0].set_ylabel("参数值")
    axs[1, 0].set_xlim(0, max_data_points)
    axs[1, 0].grid(alpha=0.3)
    axs[1, 0].set_xticks(x_range[::tick_step])
    axs[1, 0].set_xticklabels([timestamps[i] for i in x_range[::tick_step]], rotation=45, fontsize=8)

    # 子图4：状态分布
    axs[1, 1].clear()
    axs[1, 1].set_title("状态分布（最近50次）", fontsize=11)
    state_counts = [current_data["presence_state"].count(0), current_data["presence_state"].count(1), current_data["presence_state"].count(2)]
    labels = ["无人", "有人静止", "有人运动"]
    colors = ["#999999", "#FFD700", "#FF6666"]
    axs[1, 1].pie(state_counts, labels=labels, colors=colors, autopct="%1.1f%%")

    return lines


def main():
    available_ports = [p.device for p in serial.tools.list_ports.comports()]
    if not available_ports:
        print("无可用串口（检查TTL-USB连接，手册3.2章接线图）")
        return
    print("\n可用串口：")
    for i, port in enumerate(available_ports):
        print(f"  {i}: {port}")

    port_input = input(f"\n输入串口索引（0-{len(available_ports)-1}）：")
    try:
        selected_port = available_ports[int(port_input)]
    except (ValueError, IndexError):
        selected_port = available_ports[0]
        print(f"输入无效，默认用：{selected_port}")

    reader_thread = Thread(target=serial_reader, args=(selected_port,), daemon=True)
    reader_thread.start()
    print(f"\n等待雷达数据（首次接收延迟1-2秒，手册7.1章数据频率）")

    # 初始化可视化
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("R24AVD1人体存在雷达可视化", fontsize=14, fontweight="bold")

    line1, = axs[0, 1].plot([], [], 'b-', linewidth=2, label="体动幅度")
    line2, = axs[1, 0].plot([], [], 'g-', linewidth=2, label="体征参数")
    axs[0, 1].legend()
    axs[1, 0].legend()
    lines = [line1, line2]

    # 启动动画（消除缓存警告）
    ani = animation.FuncAnimation(
        fig, update_plot, fargs=(axs, lines),
        interval=1000, blit=True,
        save_count=max_data_points  # 明确缓存帧数，消除警告
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        import serial
        import matplotlib.pyplot
    except ImportError:
        print("缺依赖，执行：pip install pyserial matplotlib")
        exit(1)
    main()