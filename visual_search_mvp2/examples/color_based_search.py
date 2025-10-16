import cv2
import sys
import os
import logging

# Добавляем путь к src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.object_counter import ObjectCounter
from src.utils.image_utils import resize_image, enhance_image
from src.utils.visualization import draw_detections

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Пути к изображениям
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scene_path = os.path.join(base_dir, "visual_search_mvp", "scene.jpg")
    query_path = os.path.join(base_dir, "visual_search_mvp", "red_query.jpg")
    
    print("Загрузка изображений...")
    scene = cv2.imread(scene_path)
    query = cv2.imread(query_path)
    
    if scene is None or query is None:
        print("Ошибка загрузки изображений!")
        return
    
    # Предобработка изображений
    scene = resize_image(scene, 1200)
    query = resize_image(query, 400)
    
    # Улучшение изображений
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
    cv2.imwrite("color_based_results.jpg", output_image)
    
    # Вывод статистики
    print(f"\n=== РЕЗУЛЬТАТЫ ===")
    print(f"Найдено объектов: {results['total_found']}")
    print(f"Всего обнаружений: {results['search_stats']['total_detections']}")
    print(f"Использовано масштабов: {results['search_stats']['scales_used']}")
    
    for i, obj in enumerate(results['objects']):
        print(f"Объект {i+1}: similarity={obj['similarity']:.3f}, bbox={obj['bbox']}")
    
    # Показ результата
    scale = 0.7
    h, w = output_image.shape[:2]
    output_resized = cv2.resize(output_image, (int(w*scale), int(h*scale)))
    
    cv2.imshow("Результаты поиска по цвету и форме", output_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()