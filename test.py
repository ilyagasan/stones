import cv2

# Функция для обработки изменения положения трекбаров
def on_trackbar(value):
    global img
    # Создаем копию изображения
    hsv_image = img.copy()
    # Преобразуем изображение в цветовое пространство HSV
    hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2HSV)
    # Получаем канал насыщенности и яркости
    saturation_channel = hsv_image[:, :, 1]
    brightness_channel = hsv_image[:, :, 2]

    # Нормализуем гистограмму насыщенности
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    saturation_channel = clahe.apply(saturation_channel)

    # Применяем пороговую насыщенность темного цвета
    dark_threshold = cv2.getTrackbarPos('Dark Saturation Threshold', 'Result')
    _, dark_saturation = cv2.threshold(saturation_channel, dark_threshold, 255, cv2.THRESH_BINARY)

    # Применяем морфологическое замыкание к каналу насыщенности
    saturation_kernel_size = cv2.getTrackbarPos('Saturation Kernel Size', 'Result')
    saturation_closing = cv2.morphologyEx(dark_saturation, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (
    saturation_kernel_size, saturation_kernel_size)))

    # Применяем морфологическое замыкание к каналу яркости
    brightness_kernel_size = cv2.getTrackbarPos('Brightness Kernel Size', 'Result')
    brightness_closing = cv2.morphologyEx(brightness_channel, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,
                                                                                                         (
                                                                                                         brightness_kernel_size,
                                                                                                         brightness_kernel_size)))

    # Объединяем оба канала после морфологического замыкания
    combined_channel = cv2.bitwise_or(saturation_closing, brightness_closing)

    # Обновляем канал яркости в изображении HSV
    hsv_image[:, :, 2] = combined_channel

    # Преобразуем обратно в цветовое пространство BGR
    result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Преобразуем изображение в оттенки серого
    gray_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

    # Применяем адаптивный пороговый метод
    thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Находим контуры на бинарном изображении
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Получаем значения трекбаров для минимального и максимального размера контуров
    min_contour_area = cv2.getTrackbarPos('Min Contour Area', 'Result')
    max_contour_area = cv2.getTrackbarPos('Max Contour Area', 'Result')

    # Ограничиваем размер контуров
    filtered_contours = [cnt for cnt in contours if min_contour_area < cv2.contourArea(cnt) < max_contour_area]

    # Аппроксимируем контуры
    approx_contours = [cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True) for cnt in filtered_contours]

    # Рисуем аппроксимированные контуры на изображении
    cv2.drawContours(result_image, approx_contours, -1, (0, 255, 0), 3)

    # Отображаем результат
    cv2.imshow('Result', result_image)


# Загружаем изображение
img = cv2.imread('data/000082.png')

# Создаем окно для отображения результатов
cv2.namedWindow('Result')

# Устанавливаем начальные значения трекбаров
initial_saturation_kernel_size = 1
initial_brightness_kernel_size = 17
initial_dark_threshold = 223
initial_min_contour_area = 100
initial_max_contour_area = 10000

# Максимальные значения трекбаров
max_kernel_size = 100
max_threshold = 255

# Создаем трекбары для размеров ядра морфологического замыкания для насыщенности и яркости,
# а также для порогового значения насыщенности темного цвета
cv2.createTrackbar('Saturation Kernel Size', 'Result', initial_saturation_kernel_size, max_kernel_size, on_trackbar)
cv2.createTrackbar('Brightness Kernel Size', 'Result', initial_brightness_kernel_size, max_kernel_size, on_trackbar)
cv2.createTrackbar('Dark Saturation Threshold', 'Result', initial_dark_threshold, max_threshold, on_trackbar)

# Создаем трекбары для минимального и максимального размера контуров
cv2.createTrackbar('Min Contour Area', 'Result', initial_min_contour_area, 1000, on_trackbar)
cv2.createTrackbar('Max Contour Area', 'Result', initial_max_contour_area, 10000, on_trackbar)

# Вызываем обработчик событий, чтобы обновить изображение
on_trackbar(0)

# Ожидаем нажатие клавиши для завершения программы
cv2.waitKey(0)
cv2.destroyAllWindows()
