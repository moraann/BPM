import cv2
import numpy as np

img_file = '3.jpg'  # 待处理图片路径

def find_image(image):
    """智能裁剪出图片中的浅蓝色矩形区域并进行透视变换
        Args:
            image (Mat): 输入的BGR图像
        Returns:
            result (Mat): 标记出矩形区域的图像
            cropped_image (Mat): 裁剪并透视变换后的图像
    """
    result = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 定义浅蓝色的HSV范围
    lower_blue = np.array([90, 40, 40])    
    upper_blue = np.array([120, 255, 255]) 
    
    # 创建掩码，提取浅蓝色区域
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # 对掩码进行形态学操作，去除噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 填充内部空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 去除外部噪点
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 筛选出矩形轮廓
    rect_contour = None
    max_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:
            continue
        
        # 轮廓近似，获取多边形
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # 如果是四边形且面积最大
        if len(approx) == 4 and area > max_area:
            max_area = area
            rect_contour = approx
    
    if rect_contour is None:
        print("未找到浅蓝色矩形区域")
        return None
    
    # 绘制找到的矩形轮廓
    cv2.drawContours(result, [rect_contour], -1, (0, 255, 0), 3)
    
    # 提取矩形的四个顶点并排序
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        
        # 左上角点x+y最小，右下角点x+y最大
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # 右上角点x-y最小，左下角点x-y最大
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    points = rect_contour.reshape(4, 2)
    ordered_points = order_points(points)
    (tl, tr, br, bl) = ordered_points
    
    # 计算目标图像的宽度和高度
    width_a = np.sqrt(((br[0] - bl[0]) **2) + ((br[1] - bl[1])** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) **2) + ((tr[1] - tl[1])** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) **2) + ((tr[1] - br[1])** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) **2) + ((tl[1] - bl[1])** 2))
    max_height = max(int(height_a), int(height_b))
    
    # 定义目标图像的四个顶点
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    
    # 计算透视变换矩阵并应用
    M = cv2.getPerspectiveTransform(ordered_points, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    
    #固定裁剪
    h, w = warped.shape[:2]
    start_row = int(h * 1/9)  
    end_row = int(h)   
    start_col = 0  
    end_col = int(w * 17/24)    
    cropped_image = warped[start_row:end_row, start_col:end_col]

    return result, cropped_image


# 读取原图
image = cv2.imread(img_file, cv2.IMREAD_COLOR)
if image is None:
    print("无法读取原图文件，请检查路径是否正确")
    exit()
cv2.imshow("image", image)

# 智能裁剪
result, cropped_image = find_image(image)
if cropped_image is None:
    print("无法裁剪浅蓝色区域，请检查图像内容是否正确")
    exit()

# 灰度处理
image_gry = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
if image_gry is None:
    print("无法获取灰度图像")
    exit()
cv2.imshow("image_gry", image_gry)

# 二值化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
image_eq = clahe.apply(image_gry)
_, image_bin = cv2.threshold(image_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("image_bin", image_bin)

# 取反
image_bin_inv = cv2.bitwise_not(image_bin)

# 膨胀处理
element = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8)) 
image_dil = cv2.dilate(image_bin_inv, element)
cv2.imshow("image_dil", image_dil)

# 轮廓寻找
contours_out, hierarchy = cv2.findContours(image_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
num_location = [cv2.boundingRect(contour) for contour in contours_out]

# 过滤过小的噪声轮廓
min_area = 50  # 最小面积阈值
num_location = [loc for loc in num_location if (loc[2] * loc[3]) > min_area]  
if not num_location:
    print("未检测到有效数字区域")
    exit()

# 计算每个数字区域的y中心和高度
# 每个loc格式：(x, y, w, h)
y_centers = [y + h/2 for (x, y, w, h) in num_location]  # y中心 = y + 高度/2
heights = [h for (x, y, w, h) in num_location]
avg_height = np.mean(heights)  # 平均高度，用于确定行阈值
row_threshold = avg_height * 0.6  # 行阈值：超过此值视为不同行

# 按y中心排序，初步按垂直位置分组
sorted_by_y = sorted(zip(num_location, y_centers), key=lambda x: x[1])

# 分组为多行
rows = []
current_row = [sorted_by_y[0][0]]  # 第一行初始化为第一个元素

for i in range(1, len(sorted_by_y)):
    loc, y_center = sorted_by_y[i]
    y_diff = y_center - sorted_by_y[i-1][1]
    if y_diff < row_threshold:
        current_row.append(loc)  
    else:
        rows.append(current_row)  
        current_row = [loc]  
rows.append(current_row)  

# 每行内按x坐标排序（从左到右）
sorted_rows = []
for row in rows:
    sorted_row = sorted(row, key=lambda x: x[0])
    sorted_rows.append(sorted_row)

# 绘制轮廓
image_with_contours = cv2.cvtColor(image_bin, cv2.COLOR_GRAY2BGR)
# 遍历所有行的轮廓并绘制
all_contours = [contour for row in sorted_rows for contour in row]  # 按排序后的顺序取轮廓
# 找到轮廓对应的索引
contour_indices = []
for loc in all_contours:
    # 找到与loc匹配的轮廓索引（通过boundingRect匹配）
    for i, cnt in enumerate(contours_out):
        cnt_loc = cv2.boundingRect(cnt)
        if cnt_loc == loc:
            contour_indices.append(i)
            break
# 绘制排序后的轮廓
cv2.drawContours(image_with_contours, [contours_out[i] for i in contour_indices], -1, (0, 255, 0), 2)
cv2.imshow("Contours", image_with_contours)

# 定义判断区域是否全为白色的函数
def is_all_white(image, row_start, row_end, col_start, col_end):
    # 防止索引越界
    row_start = max(0, min(row_start, image.shape[0]-1))
    row_end = max(0, min(row_end, image.shape[0]-1))
    col_start = max(0, min(col_start, image.shape[1]-1))
    col_end = max(0, min(col_end, image.shape[1]-1))
    
    white_num = 0
    for j in range(row_start, row_end + 1):
        for i in range(col_start, col_end + 1):
            if image[j][i] == 255:
                white_num += 1
    return white_num >= 5  

# 定义穿线法识别数字的函数
def tube_identification(inputmat):
    tube = 0
    h, w = inputmat.shape
    # 调整七段管区域坐标，适应不同大小的数字
    tubo_roi = [
        [h * 0/3, h * 1/3, w * 1/2, w * 1/2],  # a
        [h * 1/3, h * 1/3, w * 2/3, w - 1],     # b
        [h * 2/3, h * 2/3, w * 2/3, w - 1],     # c
        [h * 2/3, h - 1,   w * 1/2, w * 1/2], # d
        [h * 2/3, h * 2/3, w * 0/3, w * 1/3],   # e
        [h * 1/3, h * 1/3, w * 0/3, w * 1/3],   # f
        [h * 1/3, h * 2/3, w * 1/2, w * 1/2]    # g
    ] 
    for i in range(7):
        row_start, row_end, col_start, col_end = tubo_roi[i]
        if is_all_white(inputmat, int(row_start), int(row_end), int(col_start), int(col_end)):
            tube += pow(2, i)
    
    # 特殊处理数字1（窄长）
    if h / w > 2:
        tube = 6
    # 数字映射
    num_map = {
        63: 0, 6: 1, 91: 2, 79: 3, 102: 4, 110: 4,
        109: 5, 125: 6, 39: 7, 127: 8, 111: 9
    }
    return num_map.get(tube, -1)

# 存储识别结果（按行分组）
detected_rows = []
for row in sorted_rows:
    row_numbers = []
    for loc in row:
        x, y, w, h = loc
        num_region = image_dil[y:y+h, x:x+w]  # 提取数字区域
        detected_num = tube_identification(num_region)
        row_numbers.append(detected_num)
    detected_rows.append(row_numbers)

if len(detected_rows) <= 2:
    print("未检测到多行数字，无法输出结果")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()
# 输出结果（按行显示）
print("识别结果（按行划分）：")
for i, row in enumerate(detected_rows, 1):
    print(f"第{i}行：", row)
pressure = [0,0,0]
for num in detected_rows[0]:
    if num != -1:
        pressure[0] = pressure[0]*10 + num
for num in detected_rows[1]:
    if num != -1:
        pressure[1] = pressure[1]*10 + num
for num in detected_rows[2]:
    if num != -1:
        pressure[2] = pressure[2]*10 + num
print("组合输出：")
print(pressure)
cv2.waitKey(0)
cv2.destroyAllWindows()