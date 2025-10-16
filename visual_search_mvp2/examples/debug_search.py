import cv2
import sys
import os
import logging
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.object_counter_debug import ObjectCounterDebug
from src.utils.visualization import draw_detections
from scene_generator import generate_scene, generate_query_object

# Детальная настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('search_debug.log')
    ]
)

def test_with_detailed_debug():
    """Тестирование с детальной отладкой"""
    
    print("=" * 60)
    print("ДЕТАЛЬНЫЙ ТЕСТ ПОИСКА ОБЪЕКТОВ")
    print("=" * 60)
    
    # Генерация простой сцены для теста
    print("Генерация тестовой сцены...")
    scene, objects_info = generate_scene(width=800, height=600, num_objects=8)
    
    # Создаем простой запрос - красный квадрат
    print("Создание эталонного объекта...")
    query = generate_query_object('cube', (0, 0, 255), size=100)  # Красный куб
    
    # Сохраняем исходные данные
    cv2.imwrite("debug_scene.jpg", scene)
    cv2.imwrite("debug_query.jpg", query)
    
    # Анализируем сцену
    red_objects = [obj for obj in objects_info if tuple(obj['color']) == (0, 0, 255)]
    print(f"На сцене должно быть {len(red_objects)} красных объектов")
    
    for i, obj in enumerate(red_objects):
        print(f"  Красный объект {i+1}: {obj['type']} в позиции {obj['position']}")
    
    # Инициализация счетчика с либеральными параметрами
    print("Инициализация системы поиска...")
    counter = ObjectCounterDebug(color_tolerance=100, min_similarity=0.3)
    
    # Запуск поиска с отладкой
    print("Запуск поиска с отладкой...")
    results = counter.count_objects_debug(scene, query, "detailed_debug")
    
    # Визуализация результатов
    output_image = draw_detections(scene, results['objects'], query)
    cv2.imwrite("debug_results.jpg", output_image)
    
    # Детальная статистика
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ПОИСКА:")
    print("=" * 60)
    print(f"Цветовое покрытие сцены: {results['color_coverage']:.2f}%")
    print(f"Найдено объектов: {results['total_found']}")
    print(f"Ожидалось красных объектов: {len(red_objects)}")
    
    if red_objects:
        accuracy = results['total_found'] / len(red_objects) * 100
        print(f"Точность: {accuracy:.1f}%")
    
    print(f"\nДетали найденных объектов:")
    for i, obj in enumerate(results['objects']):
        bbox = obj['bbox']
        similarity = obj['similarity']
        print(f"  Объект {i+1}: similarity={similarity:.3f}, bbox=({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})")
    
    # Показываем все отладочные изображения
    print(f"\nОтладочные изображения сохранены:")
    print(f"  - debug_scene.jpg (исходная сцена)")
    print(f"  - debug_query.jpg (эталонный объект)") 
    print(f"  - debug_results.jpg (результаты поиска)")
    print(f"  - detailed_debug_query.jpg (анализ запроса)")
    print(f"  - detailed_debug_color_coverage.jpg (цветовое покрытие)")
    
    # Показ результата
    scale = 0.8
    h, w = output_image.shape[:2]
    output_resized = cv2.resize(output_image, (int(w*scale), int(h*scale)))
    
    cv2.imshow("Результаты поиска (Детальный отладка)", output_resized)
    print("\nНажмите любую клавишу чтобы закрыть окно...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_simple_test():
    """Создание простого теста с гарантированным результатом"""
    print("Создание простого теста...")
    
    # Создаем простую сцену с одним красным квадратом
    scene = np.ones((400, 400, 3), dtype=np.uint8) * 150  # Серый фон
    
    # Добавляем красный квадрат
    cv2.rectangle(scene, (100, 100), (200, 200), (0, 0, 255), -1)
    
    # Создаем запрос - красный квадрат
    query = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(query, (10, 10), (90, 90), (0, 0, 255), -1)
    
    cv2.imwrite("simple_scene.jpg", scene)
    cv2.imwrite("simple_query.jpg", query)
    
    print("Простой тест создан!")
    print("На сцене: 1 красный квадрат")
    print("Запрос: красный квадрат")
    
    return scene, query

def test_simple_case():
    """Тестирование на простом случае"""
    print("ТЕСТ НА ПРОСТОМ СЛУЧАЕ")
    
    scene, query = create_simple_test()
    
    counter = ObjectCounterDebug(color_tolerance=80, min_similarity=0.3)
    results = counter.count_objects_debug(scene, query, "simple_debug")
    
    output_image = draw_detections(scene, results['objects'], query)
    cv2.imwrite("simple_results.jpg", output_image)
    
    print(f"Найдено объектов: {results['total_found']} (ожидается: 1)")
    
    if results['total_found'] == 1:
        print("✅ ТЕСТ ПРОЙДЕН! Система работает корректно.")
    else:
        print("❌ ТЕСТ НЕ ПРОЙДЕН! Есть проблемы в алгоритме.")
    
    cv2.imshow("Простой тест", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Выберите тест:")
    print("1 - Детальный тест на сгенерированной сцене")
    print("2 - Простой гарантированный тест")
    
    choice = input("Введите номер: ").strip()
    
    if choice == "1":
        test_with_detailed_debug()
    elif choice == "2":
        test_simple_case()
    else:
        print("Запускаю простой тест...")
        test_simple_case()