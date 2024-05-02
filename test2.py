import cv2
import numpy as np


# Функция для обработки изменения положения трекбаров
def on_trackbar(value):
    global img
    # Создаем копию изображения
    hsv_image = img.copy()
    # Преобразуем изображение в цветовое пространство HSV
    hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2HSV)

    # Определяем цветовой диапазон для поиска
    lower_color = np.array([20, 30, 210])  # Нижний порог для цвета #FFEFB1 в цветовом пространстве HSV
    upper_color = np.array([40, 50, 255])  # Верхний порог для цвета #FFEFB1 в цветовом пространстве HSV

    # Создаем маску для нахождения областей, близких к указанному цвету
    mask = cv2.inRange(hsv_image, lower_color, upper_color)

    # Находим контуры на бинарном изображении
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Рисуем контуры на исходном изображении
    result_image = img.copy()
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 3)

    # Отображаем результат
    cv2.imshow('Result', result_image)


# Загружаем изображение
img = cv2.imread('data/000098.png')

# Создаем окно для отображения результатов
cv2.namedWindow('Result')

# Устанавливаем начальные значения трекбаров

# Вызываем обработчик событий, чтобы обновить изображение
on_trackbar(0)

# Ожидаем нажатие клавиши для завершения программы
cv2.waitKey(0)
cv2.destroyAllWindows()
