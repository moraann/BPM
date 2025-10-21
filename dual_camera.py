import cv2
import time
import threading
import os


class DualCameraViewer:
    def __init__(self):
        # 初始化摄像头
        self.thermal_cap = cv2.VideoCapture(0)  # 热成像摄像头
        self.visible_cap = cv2.VideoCapture(1)  # 可见光摄像头
        if not self.thermal_cap.isOpened():
            raise IOError("无法打开热成像摄像头，请检查连接")
        if not self.visible_cap.isOpened():
            raise IOError("无法打开可见光摄像头，请检查连接")
        
        # 获取热成像摄像头参数（需要截取上半部分）
        self.thermal_width = int(self.thermal_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.thermal_height = int(self.thermal_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.thermal_top_half_height = self.thermal_height // 2  # 上半部分高度
        
        # 获取可见光摄像头参数（需要完整显示）
        self.visible_width = int(self.visible_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.visible_height = int(self.visible_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 视频保存参数
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.recording = False
        self.out_thermal = None  # 热成像视频写入器
        self.out_visible = None  # 可见光视频写入器
        self.record_duration = 10  # 录制时长（秒）
        self.output_dir = "camera_recordings"  # 录像保存目录
        
        # 若不存在则创建保存目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 显示信息
        self.info_text = "SPACE"
        self.thermal_name = "Thermal"
        self.visible_name = "VL"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 热成像显示模式控制
        self.show_thermal_top_half = True
        
        # 打印调试信息
        print(f"热成像原始分辨率: {self.thermal_width}x{self.thermal_height}，显示上半部分: {self.thermal_width}x{self.thermal_top_half_height}")
        print(f"可见光分辨率: {self.visible_width}x{self.visible_height}（完整显示）")
        
    def start_recording(self):
        if self.recording:
            return 
        self.recording = True
        
        # 生成文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")  # 时间戳
        thermal_filename = os.path.join(self.output_dir, f"thermal_{timestamp}.mp4")
        visible_filename = os.path.join(self.output_dir, f"visible_{timestamp}.mp4")
        
        # 获取帧率（使用热成像的帧率，如无法获取则用25fps）
        fps = int(self.thermal_cap.get(cv2.CAP_PROP_FPS)) or 25
        
        # 创建视频写入器（分别使用各自的尺寸）
        self.out_thermal = cv2.VideoWriter(
            thermal_filename, 
            self.fourcc, 
            fps, 
            (self.thermal_width, self.thermal_top_half_height)  # 热成像用截取后的尺寸
        )
        
        self.out_visible = cv2.VideoWriter(
            visible_filename, 
            self.fourcc, 
            fps, 
            (self.visible_width, self.visible_height)  # 可见光用完整尺寸
        )
        
        # 录制计时
        start_time = time.time()
        self.info_text = f"{self.record_duration}Sec"
        
        while self.recording and (time.time() - start_time) < self.record_duration:
            elapsed = time.time() - start_time
            remaining = self.record_duration - int(elapsed)
            self.info_text = f"{remaining}Sec"
            time.sleep(0.1)
            
        # 结束录制
        self.stop_recording()
        self.info_text = "Success"
        print(f"热成像视频保存至: {thermal_filename}")
        print(f"可见光视频保存至: {visible_filename}")
        
    def stop_recording(self):
        """停止录制并释放资源"""
        if self.recording:
            self.recording = False
            if self.out_thermal is not None:
                self.out_thermal.release()
                self.out_thermal = None
            if self.out_visible is not None:
                self.out_visible.release()
                self.out_visible = None
    
    def run(self):
        print("程序已启动。按空格键开始录制10秒视频,按T键切换热成像显示模式,按ESC键退出")
        cv2.namedWindow('Dual Camera', cv2.WINDOW_NORMAL)
        try:
            while True:
                # 读取两个摄像头的帧
                ret_thermal, frame_thermal = self.thermal_cap.read()
                ret_visible, frame_visible = self.visible_cap.read()
                
                # 检查帧是否读取成功
                if not ret_thermal:
                    print("无法获取热成像画面，程序将退出")
                    break
                if not ret_visible:
                    print("无法获取可见光画面，程序将退出")
                    break
                
                # 处理热成像画面（截取上半部分）
                if self.show_thermal_top_half:
                    frame_thermal = frame_thermal[:self.thermal_top_half_height, :]
                
                # 如果正在录制，写入视频文件
                if self.recording:
                    if self.out_thermal is not None:
                        self.out_thermal.write(frame_thermal)
                    if self.out_visible is not None:
                        self.out_visible.write(frame_visible)
                    
                    # 绘制录制指示（红色圆点）
                    cv2.circle(frame_thermal, (20, 20), 10, (0, 0, 255), -1)
                    cv2.circle(frame_visible, (20, 20), 10, (0, 0, 255), -1)
                
                # 在画面上添加文字信息
                # 热成像画面
                cv2.putText(frame_thermal, self.thermal_name, (10, 25),
                           self.font, 0.7, (0, 255, 0), 2)
                # 可见光画面
                cv2.putText(frame_visible, self.visible_name, (10, 25),
                           self.font, 0.7, (0, 255, 0), 2)
                # 底部提示信息
                cv2.putText(frame_thermal, self.info_text, (10, frame_thermal.shape[0] - 10),
                           self.font, 0.6, (0, 255, 255), 2)
                
                # 调整两个画面的高度一致，方便横向拼接
                target_height = max(frame_thermal.shape[0], frame_visible.shape[0])
                frame_thermal_resized = cv2.resize(frame_thermal, 
                                                  (int(frame_thermal.shape[1] * target_height / frame_thermal.shape[0]), 
                                                   target_height))
                frame_visible_resized = cv2.resize(frame_visible, 
                                                  (int(frame_visible.shape[1] * target_height / frame_visible.shape[0]), 
                                                   target_height))
                
                # 横向拼接并显示
                combined_frame = cv2.hconcat([frame_thermal_resized, frame_visible_resized])
                cv2.imshow('Dual Camera', combined_frame)
                
                # 处理操控信息
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键退出
                    break
                elif key == ord(' '):  # 空格键开始录制
                    if not self.recording:
                        threading.Thread(target=self.start_recording, daemon=True).start()
                elif key == ord('t'):  # T键切换热成像显示模式
                    self.show_thermal_top_half = not self.show_thermal_top_half
                    mode = "Top" if self.show_thermal_top_half else "All"
                    self.info_text = f"{mode}"
        
        finally:
            self.stop_recording()
            self.thermal_cap.release()
            self.visible_cap.release()
            cv2.destroyAllWindows()
            print("程序已退出")

if __name__ == "__main__":
    try:
        viewer = DualCameraViewer()
        viewer.run()
    except Exception as e:
        print(f"发生错误: {str(e)}")