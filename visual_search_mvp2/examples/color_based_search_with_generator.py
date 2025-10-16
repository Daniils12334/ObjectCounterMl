import cv2
import sys
import os
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.object_counter import ObjectCounter
from src.utils.image_utils import resize_image, enhance_image
from src.utils.visualization import draw_detections
from scene_generator import generate_scene, generate_query_object

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_with_generated_scene():
    """Тестирование на сгенерированной сцене"""
    
    # Генерация сцены
    print("Генерация тестовой сцены...")
    scene, objects_info = generate_scene(width=1400, height=1000, num_objects=15)
    
    # Выбираем целевой объект (красные звезды)
    target_shape = 'star'
    target_color = (0, 0, 255)  # Красный
    
    # Считаем сколько таких объектов должно быть найдено
    expected_count = sum(1 for obj in objects_info 
                        if obj['type'] == target_shape and obj['color'] == target_color)
    
    print(f"Ожидается найти: {expected_count} {target_shape} объектов цвета {target_color}")
    
    # Генерация эталонного изображения
    query = generate_query_object(target_shape, target_color, size=180)
    
    # Сохраняем для отладки
    cv2.imwrite("generated_scene.jpg", scene)
    cv2.imwrite("generated_query.jpg", query)
    
    # Инициализация счетчика объектов
    counter = ObjectCounter(color_tolerance=70, min_similarity=0.5)
    
    # Поиск объектов
    print("Запуск поиска...")
    results = counter.count_objects(scene, query)
    
    # Визуализация результатов
    output_image = draw_detections(scene, results['objects'], query)
    
    # Сохранение результата
    cv2.imwrite("generated_scene_results.jpg", output_image)
    
    # Вывод статистики
    print(f"\n=== РЕЗУЛЬТАТЫ ===")
    print(f"Найдено объектов: {results['total_found']}")
    print(f"Ожидалось: {expected_count}")
    print(f"Точность: {results['total_found'] / expected_count * 100:.1f}%" if expected_count > 0 else "N/A")
    print(f"Всего обнаружений: {results['search_stats']['total_detections']}")
    print(f"Использовано масштабов: {results['search_stats']['scales_used']}")
    
    for i, obj in enumerate(results['objects']):
        print(f"Объект {i+1}: similarity={obj['similarity']:.3f}, bbox={obj['bbox']}")
    
    # Показ результата
    scale = 0.7
    h, w = output_image.shape[:2]
    output_resized = cv2.resize(output_image, (int(w*scale), int(h*scale)))
    
    cv2.imshow("Результаты поиска на сгенерированной сцене", output_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_with_real_images():
    """Тестирование на реальных изображениях"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scene_path = os.path.join(base_dir, "visual_search_mvp", "scene.jpg")
    query_path = os.path.join(base_dir, "visual_search_mvp", "red_query.jpg")
    
    print("Загрузка реальных изображений...")
    scene = cv2.imread(scene_path)
    query = cv2.imread(query_path)
    
    if scene is None or query is None:
        print("Ошибка загрузки изображений!")
        return
    
    # Предобработка изображений
    scene = resize_image(scene, 1200)
    query = resize_image(query, 400)
    scene = enhance_image(scene)
    query = enhance_image(query)
    
    print(f"Сцена: {scene.shape}")
    print(f"Запрос: {query.shape}")
    
    # Инициализация счетчика объектов
    counter = ObjectCounter(color_tolerance=60, min_similarity=0.5)
    
    # Поиск объектов
    print("Запуск поиска...")
    results = counter.count_objects(scene, query)
    
    # Визуализация результатов
    output_image = draw_detections(scene, results['objects'], query)
    
    # Сохранение результата
    cv2.imwrite("real_images_results.jpg", output_image)
    
    # Вывод статистики
    print(f"\n=== РЕЗУЛЬТАТЫ ===")
    print(f"Найдено объектов: {results['total_found']}")
    
    for i, obj in enumerate(results['objects']):
        print(f"Объект {i+1}: similarity={obj['similarity']:.3f}, bbox={obj['bbox']}")
    
    # Показ результата
    scale = 0.7
    h, w = output_image.shape[:2]
    output_resized = cv2.resize(output_image, (int(w*scale), int(h*scale)))
    
    cv2.imshow("Результаты поиска на реальных изображениях", output_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Выберите режим тестирования:")
    print("1 - Сгенерированная сцена")
    print("2 - Реальные изображения")
    
    choice = input("Введите номер (1 или 2): ").strip()
    
    if choice == "1":
        test_with_generated_scene()
    elif choice == "2":
        test_with_real_images()
    else:
        print("Неверный выбор. Запускаю тест на сгенерированной сцене...")
        test_with_generated_scene()