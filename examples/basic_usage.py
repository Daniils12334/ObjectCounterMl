import cv2
import sys
import os
import numpy as np

# Добавляем путь к src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.segmenters import Segmenter
from src.core.embedders import Embedder
from src.core.search_engine import VisualSearchEngine
from src.utils.visualization import draw_results

def debug_segmentation(segmenter, image, output_path="debug_segmentation.jpg"):
    """Функция для отладки сегментации"""
    print("=== ОТЛАДКА СЕГМЕНТАЦИИ ===")
    
    # Сегментируем объекты
    objects = segmenter.segment(image)
    print(f"Найдено объектов: {len(objects)}")
    
    # Создаем изображение для визуализации
    debug_image = image.copy()
    
    for i, obj in enumerate(objects):
        bbox = obj['bbox']
        class_name = obj['class_name']
        score = obj['score']
        
        # Рисуем bounding box
        x, y, w, h = bbox
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Добавляем текст
        label = f"{class_name}: {score:.2f}"
        cv2.putText(debug_image, label, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        print(f"Объект {i}: {class_name} (confidence: {score:.3f}), bbox: {bbox}")
    
    # Сохраняем результат
    cv2.imwrite(output_path, debug_image)
    print(f"Результат сегментации сохранен в: {output_path}")
    
    return objects

def main():
    print("Загрузка изображений...")
    
    # Пути к изображениям - используем абсолютные пути
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scene_path = os.path.join(base_dir, "visual_search_mvp", "scene.jpg")
    query_path = os.path.join(base_dir, "visual_search_mvp", "red_query.jpg")
    
    print(f"Сцена: {scene_path}")
    print(f"Запрос: {query_path}")
    
    # Проверяем существование файлов
    if not os.path.exists(scene_path):
        print(f"ОШИБКА: Файл сцены не найден: {scene_path}")
        return
        
    if not os.path.exists(query_path):
        print(f"ОШИБКА: Файл запроса не найден: {query_path}")
        return
    
    # Загрузка изображений
    scene = cv2.imread(scene_path)
    query = cv2.imread(query_path)
    
    if scene is None:
        print("ОШИБКА: Не удалось загрузить изображение сцены")
        return
        
    if query is None:
        print("ОШИБКА: Не удалось загрузить изображение запроса")
        return
    
    print(f"Размер сцены: {scene.shape}")
    print(f"Размер запроса: {query.shape}")
    
    # Инициализация системы
    print("Инициализация моделей...")
    segmenter = Segmenter(confidence_threshold=0.5)  # Понижаем порог для теста
    embedder = Embedder()
    
    # ОТЛАДКА: Проверяем сегментацию
    objects = debug_segmentation(segmenter, scene, "segmentation_debug.jpg")
    
    if not objects:
        print("ВНИМАНИЕ: Сегментатор не нашел ни одного объекта!")
        print("Попробуйте:")
        print("1. Уменьшить confidence_threshold")
        print("2. Проверить что на изображении есть заметные объекты")
        print("3. Попробовать другое изображение")
        return
    
    # Инициализация поискового движка с низким порогом для теста
    search_engine = VisualSearchEngine(segmenter, embedder, similarity_threshold=0.6)
    
    # Поиск похожих объектов
    print("Запуск поиска похожих объектов...")
    results = search_engine.search_similar_objects(scene, query)
    
    print(f"Всего объектов на сцене: {results['total_objects']}")
    print(f"Найдено похожих объектов: {len(results['matches'])}")
    
    # Визуализация
    output = draw_results(scene, results['matches'], query)
    
    # Сохранение результата
    cv2.imwrite("search_results.jpg", output)
    
    # Показываем результат
    scale = 0.6
    h, w = output.shape[:2]
    output_resized = cv2.resize(output, (int(w*scale), int(h*scale)))
    
    cv2.imshow("Результаты поиска", output_resized)
    print("Нажмите любую клавишу чтобы закрыть окно...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Выводим детальную информацию о найденных объектах
    if results['matches']:
        print("\n=== НАЙДЕННЫЕ ОБЪЕКТЫ ===")
        for i, match in enumerate(results['matches']):
            print(f"Объект {i+1}: similarity = {match['similarity']:.3f}, bbox = {match['bbox']}")
    else:
        print("\nПохожие объекты не найдены.")
        print("Рекомендации:")
        print("1. Уменьшить similarity_threshold (текущий: 0.6)")
        print("2. Убедиться что query изображение похоже на искомые объекты")
        print("3. Проверить что сегментатор находит нужные объекты")

if __name__ == "__main__":
    main()