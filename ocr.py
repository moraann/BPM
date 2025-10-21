import cv2
import numpy as np

img_file = '2.jpg'
# 原图
image = cv2.imread(img_file, cv2.IMREAD_COLOR)
cv2.imshow("image", image)

# 灰度图处理
image_gry = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
if image_gry is None:
    print("无法读取图片")
    exit()
cv2.imshow("image_gry", image_gry)

# 二值化
_, image_bin = cv2.threshold(image_gry, 200, 255, cv2.THRESH_BINARY)
cv2.imshow("image_bin", image_bin)

image_bin_inv = cv2.bitwise_not(image_bin)

# 进行膨胀处理（调整结构元素大小，避免过度膨胀）
element = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # 缩小结构元素，更适合细分数字
image_dil = cv2.dilate(image_bin_inv, element)
cv2.imshow("image_dil", image_dil)

# 轮廓寻找
contours_out, hierarchy = cv2.findContours(image_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 简化轮廓点
num_location = [cv2.boundingRect(contour) for contour in contours_out]

# -------------------------- 关键改进：行分组与排序 --------------------------
# 1. 过滤过小的噪声轮廓（避免干扰行判断）
min_area = 50  # 根据实际数字大小调整，过滤小噪声
num_location = [loc for loc in num_location if (loc[2] * loc[3]) > min_area]  # w*h > 最小面积

if not num_location:
    print("未检测到有效数字区域")
    exit()

# 2. 计算每个数字区域的y中心（用于判断行）和高度（用于确定行阈值）
# 每个loc格式：(x, y, w, h)
y_centers = [y + h/2 for (x, y, w, h) in num_location]  # y中心 = y + 高度/2
heights = [h for (x, y, w, h) in num_location]
avg_height = np.mean(heights)  # 平均高度，用于确定行阈值
row_threshold = avg_height * 0.6  # 行阈值：超过此值视为不同行（可根据实际调整）

# 3. 按y中心排序，初步按垂直位置分组
# 将位置信息与y中心绑定，按y中心排序
sorted_by_y = sorted(zip(num_location, y_centers), key=lambda x: x[1])

# 4. 分组为多行
rows = []
current_row = [sorted_by_y[0][0]]  # 第一行初始化为第一个元素

for i in range(1, len(sorted_by_y)):
    loc, y_center = sorted_by_y[i]
    # 计算与上一个元素的y中心差
    y_diff = y_center - sorted_by_y[i-1][1]
    if y_diff < row_threshold:
        current_row.append(loc)  # 同一行
    else:
        rows.append(current_row)  # 存入上一行
        current_row = [loc]  # 新行
rows.append(current_row)  # 存入最后一行

# 5. 每行内按x坐标排序（从左到右）
sorted_rows = []
for row in rows:
    # 按x坐标排序
    sorted_row = sorted(row, key=lambda x: x[0])
    sorted_rows.append(sorted_row)

# --------------------------------------------------------------------------

# 绘制轮廓（修正拼写错误：inmage → image）
image_with_contours = cv2.cvtColor(image_bin, cv2.COLOR_GRAY2BGR)
# 遍历所有行的轮廓并绘制
all_contours = [contour for row in sorted_rows for contour in row]  # 按排序后的顺序取轮廓
# 找到轮廓对应的索引（因为contours_out与num_location顺序一致）
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
    return white_num >= 5  # 调整阈值，适应实际图像

# 定义穿线法识别数字的函数（保持不变）
def tube_identification(inputmat):
    tube = 0
    h, w = inputmat.shape
    # 调整七段管区域坐标，适应不同大小的数字（用比例计算更鲁棒）
    tubo_roi = [
        [h * 0/3, h * 1/3, w * 1/2, w * 1/2],  # a
        [h * 1/3, h * 1/3, w * 2/3, w - 1],     # b
        [h * 2/3, h * 2/3, w * 2/3, w - 1],     # c
        [h * 2/3, h - 1,     w * 1/2, w * 1/2], # d
        [h * 2/3, h * 2/3, w * 0/3, w * 1/3],   # e
        [h * 1/3, h * 1/3, w * 0/3, w * 1/3],   # f
        [h * 1/3, h * 2/3, w * 1/2, w * 1/2]    # g
    ] 
    for i in range(7):
        row_start, row_end, col_start, col_end = tubo_roi[i]
        if is_all_white(inputmat, int(row_start), int(row_end), int(col_start), int(col_end)):
            tube += pow(2, i)
        # 绘制检测线（可选）
        #cv2.line(inputmat, (int(col_end), int(row_end), (int(col_start), int(row_start), (255, 0, 0), 1)))
    
    # 特殊处理数字1（窄长）
    if h / w > 2:
        tube = 6
    # 数字映射
    num_map = {
        63: 0, 6: 1, 91: 2, 79: 3, 102: 4, 110: 4,
        109: 5, 125: 6, 7: 7, 127: 8, 111: 9
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

if len(detected_rows) <= 1:
    print("未检测到多行数字，无法输出结果")
    cv2.destroyAllWindows()
    exit()
# 输出结果（按行显示）
print("识别结果（按行划分）：")
for i, row in enumerate(detected_rows, 1):
    print(f"第{i}行：", row)
pressure = [0,0]
for num in detected_rows[0]:
    if num != -1:
        pressure[0] = pressure[0]*10 + num
for num in detected_rows[1]:
    if num != -1:
        pressure[1] = pressure[1]*10 + num
print("组合输出：")
print(pressure)
cv2.waitKey(0)
cv2.destroyAllWindows()