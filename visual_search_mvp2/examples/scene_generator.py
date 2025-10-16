import cv2
import numpy as np
import random
import math
from typing import List, Dict, Tuple

def generate_star(size: int, color: Tuple[int, int, int], num_points: int = 5) -> np.ndarray:
    """
    Генерация изображения звезды
    
    Args:
        size: размер изображения
        color: цвет в BGR
        num_points: количество лучей
        
    Returns:
        Изображение звезды
    """
    image = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    
    # Параметры звезды
    outer_radius = size // 2 - 5
    inner_radius = outer_radius // 2
    
    points = []
    for i in range(num_points * 2):
        angle = math.pi / num_points * i
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = center[0] + int(radius * math.cos(angle))
        y = center[1] + int(radius * math.sin(angle))
        points.append((x, y))
    
    # Рисуем звезду
    cv2.fillPoly(image, [np.array(points)], color)
    
    return image

def generate_cube(size: int, color: Tuple[int, int, int]) -> np.ndarray:
    """
    Генерация изображения куба
    
    Args:
        size: размер изображения
        color: цвет в BGR
        
    Returns:
        Изображение куба
    """
    image = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Основная грань
    face_size = int(size * 0.7)
    margin = (size - face_size) // 2
    
    # Рисуем основную грань
    cv2.rectangle(image, (margin, margin), (margin + face_size, margin + face_size), color, -1)
    
    # Рисуем боковые грани (для 3D эффекта)
    side_width = int(face_size * 0.3)
    top_color = tuple(max(0, c - 30) for c in color)  # темнее
    side_color = tuple(max(0, c - 60) for c in color)  # еще темнее
    
    # Верхняя грань
    top_points = np.array([
        [margin, margin],
        [margin + side_width, margin - side_width],
        [margin + face_size + side_width, margin - side_width],
        [margin + face_size, margin]
    ])
    cv2.fillPoly(image, [top_points], top_color)
    
    # Боковая грань
    side_points = np.array([
        [margin + face_size, margin],
        [margin + face_size + side_width, margin - side_width],
        [margin + face_size + side_width, margin + face_size - side_width],
        [margin + face_size, margin + face_size]
    ])
    cv2.fillPoly(image, [side_points], side_color)
    
    return image

def generate_circle(size: int, color: Tuple[int, int, int]) -> np.ndarray:
    """
    Генерация изображения круга
    
    Args:
        size: размер изображения
        color: цвет в BGR
        
    Returns:
        Изображение круга
    """
    image = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = size // 2 - 5
    
    cv2.circle(image, center, radius, color, -1)
    return image

def generate_triangle(size: int, color: Tuple[int, int, int]) -> np.ndarray:
    """
    Генерация изображения треугольника
    
    Args:
        size: размер изображения
        color: цвет в BGR
        
    Returns:
        Изображение треугольника
    """
    image = np.zeros((size, size, 3), dtype=np.uint8)
    
    height = int(size * 0.9)
    points = np.array([
        [size // 2, 5],  # верхняя точка
        [5, height],     # левая нижняя
        [size - 5, height]  # правая нижняя
    ])
    
    cv2.fillPoly(image, [points], color)
    return image

def generate_scene(width: int = 1600, height: int = 1200, num_objects: int = 20) -> Tuple[np.ndarray, List[Dict]]:
    """
    Генерация сложной сцены с различными объектами
    
    Args:
        width: ширина сцены
        height: высота сцены
        num_objects: количество объектов
        
    Returns:
        (изображение сцены, список информации об объектах)
    """
    # Создаем фон с градиентом
    scene = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Добавляем градиентный фон
    for y in range(height):
        color_val = int(100 + 50 * math.sin(y / 100))
        scene[y, :] = (color_val, color_val, color_val)
    
    # Добавляем случайный шум для текстуры
    noise = np.random.randint(0, 10, (height, width, 3), dtype=np.uint8)
    scene = cv2.add(scene, noise)
    
    # Доступные формы и цвета
    shapes = ['star', 'cube', 'circle', 'triangle']
    colors = [
        (0, 0, 255),    # Красный
        (0, 255, 0),    # Зеленый  
        (255, 0, 0),    # Синий
        (0, 255, 255),  # Желтый
        (255, 0, 255),  # Пурпурный
        (255, 255, 0),  # Голубой
    ]
    
    objects_info = []
    
    for i in range(num_objects):
        # Случайные параметры
        shape_type = random.choice(shapes)
        color = random.choice(colors)
        size = random.randint(30, 120)
        
        # Случайная позиция (учитывая границы)
        x = random.randint(0, width - size - 1)
        y = random.randint(0, height - size - 1)
        
        # Случайный поворот и масштаб
        rotation = random.uniform(0, 360)
        scale = random.uniform(0.8, 1.2)
        
        # Генерация объекта
        if shape_type == 'star':
            obj_image = generate_star(size, color)
        elif shape_type == 'cube':
            obj_image = generate_cube(size, color)
        elif shape_type == 'circle':
            obj_image = generate_circle(size, color)
        elif shape_type == 'triangle':
            obj_image = generate_triangle(size, color)
        
        # Применяем аффинные преобразования
        if rotation != 0 or scale != 1.0:
            center = (size // 2, size // 2)
            matrix = cv2.getRotationMatrix2D(center, rotation, scale)
            obj_image = cv2.warpAffine(obj_image, matrix, (size, size))
        
        # Накладываем объект на сцену (простое наложение)
        roi = scene[y:y+size, x:x+size]
        mask = np.any(obj_image > 0, axis=2)
        roi[mask] = obj_image[mask]
        
        # Сохраняем информацию об объекте
        objects_info.append({
            'type': shape_type,
            'color': color,
            'position': (x, y),
            'size': size,
            'rotation': rotation,
            'scale': scale
        })
    
    return scene, objects_info

def generate_query_object(shape_type: str, color: Tuple[int, int, int], size: int = 200) -> np.ndarray:
    """
    Генерация эталонного объекта для поиска
    
    Args:
        shape_type: тип формы
        color: цвет
        size: размер
        
    Returns:
        Изображение эталонного объекта
    """
    if shape_type == 'star':
        return generate_star(size, color)
    elif shape_type == 'cube':
        return generate_cube(size, color)
    elif shape_type == 'circle':
        return generate_circle(size, color)
    elif shape_type == 'triangle':
        return generate_triangle(size, color)
    else:
        return generate_cube(size, color)  # по умолчанию

def create_test_dataset():
    """Создание тестового набора данных"""
    
    # Генерация сцены
    print("Генерация сцены...")
    scene, objects_info = generate_scene(num_objects=25)
    cv2.imwrite("test_scene.jpg", scene)
    
    # Сохраняем информацию об объектах
    with open("scene_objects_info.txt", "w") as f:
        for i, obj in enumerate(objects_info):
            f.write(f"Object {i}: {obj}\n")
    
    # Генерация нескольких эталонных объектов
    print("Генерация эталонных объектов...")
    
    # Красные звезды
    red_star = generate_query_object('star', (0, 0, 255), 150)
    cv2.imwrite("query_red_star.jpg", red_star)
    
    # Синие кубы
    blue_cube = generate_query_object('cube', (255, 0, 0), 150)  
    cv2.imwrite("query_blue_cube.jpg", blue_cube)
    
    # Зеленые круги
    green_circle = generate_query_object('circle', (0, 255, 0), 150)
    cv2.imwrite("query_green_circle.jpg", green_circle)
    
    print("Тестовый набор данных создан!")
    print(f"Сцена: test_scene.jpg ({scene.shape[1]}x{scene.shape[0]})")
    print(f"Эталоны: query_red_star.jpg, query_blue_cube.jpg, query_green_circle.jpg")
    
    # Статистика по объектам на сцене
    from collections import Counter
    shape_counter = Counter(obj['type'] for obj in objects_info)
    color_counter = Counter(tuple(obj['color']) for obj in objects_info)
    
    print("\nСтатистика сцены:")
    print(f"Всего объектов: {len(objects_info)}")
    print("Распределение по формам:", dict(shape_counter))
    print("Распределение по цветам:", {str(k): v for k, v in color_counter.items()})

if __name__ == "__main__":
    create_test_dataset()