import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.working_counter import WorkingCounter

def create_perfect_test():
    """Создание идеального теста"""
    print("Создание идеального теста...")
    
    # Создаем сцену с несколькими красными объектами
    scene = np.ones((500, 500, 3), dtype=np.uint8) * 100  # Серый фон
    
    # Добавляем красные объекты разных размеров
    cv2.rectangle(scene, (50, 50), (150, 150), (0, 0, 255), -1)    # Большой красный квадрат
    cv2.rectangle(scene, (200, 200), (250, 250), (0, 0, 255), -1)  # Средний красный квадрат
    cv2.rectangle(scene, (300, 300), (330, 330), (0, 0, 255), -1)  # Маленький красный квадрат
    cv2.circle(scene, (400, 100), 30, (0, 0, 255), -1)             # Красный круг
    
    # Добавляем другие цвета для проверки
    cv2.rectangle(scene, (50, 300), (100, 350), (0, 255, 0), -1)   # Зеленый квадрат
    cv2.rectangle(scene, (350, 350), (400, 400), (255, 0, 0), -1)  # Синий квадрат
    
    # Создаем запрос - чистый красный квадрат
    query = np.zeros((80, 80, 3), dtype=np.uint8)
    query[:, :] = (0, 0, 255)  # Полностью красный
    
    cv2.imwrite("perfect_scene.jpg", scene)
    cv2.imwrite("perfect_query.jpg", query)
    
    print("Идеальный тест создан!")
    print("На сцене: 4 красных объекта (3 квадрата, 1 круг)")
    print("Запрос: чистый красный цвет")
    
    return scene, query

def test_perfect_case():
    """Тестирование на идеальном случае"""
    print("=" * 60)
    print("ТЕСТ НА ИДЕАЛЬНОМ СЛУЧАЕ")
    print("=" * 60)
    
    scene, query = create_perfect_test()
    
    # Инициализация счетчика
    counter = WorkingCounter(color_tolerance=100, min_similarity=0.2)
    
    # Поиск объектов
    results = counter.count_objects_working(scene, query)
    
    # Визуализация результатов
    output = scene.copy()
    for obj in results['objects']:
        x, y, w, h = obj['bbox']
        similarity = obj['similarity']
        
        # Рисуем bounding box
        color = (0, 255, 0) if similarity > 0.5 else (0, 255, 255)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        
        # Подписываем сходство
        label = f"{similarity:.2f}"
        cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Добавляем запрос в угол
    h_query, w_query = query.shape[:2]
    output[10:10+h_query, 10:10+w_query] = query
    cv2.rectangle(output, (10, 10), (10+w_query, 10+h_query), (0, 255, 0), 2)
    
    cv2.imwrite("perfect_results.jpg", output)
    
    # Вывод результатов
    print(f"\nРЕЗУЛЬТАТЫ:")
    print(f"Найдено объектов: {results['total_found']}")
    print(f"Ожидалось: 4 красных объекта")
    
    for i, obj in enumerate(results['objects']):
        bbox = obj['bbox']
        similarity = obj['similarity']
        print(f"  Объект {i+1}: similarity={similarity:.3f}, bbox=({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})")
    
    if results['total_found'] >= 3:
        print("✅ ТЕСТ ПРОЙДЕН! Система работает!")
    else:
        print("❌ ТЕСТ НЕ ПРОЙДЕН!")
    
    # Показываем результат
    cv2.imshow("Идеальный тест - результаты", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_with_real_query():
    """Тестирование с реальным query изображением"""
    print("\n" + "=" * 60)
    print("ТЕСТ С РЕАЛЬНЫМИ ИЗОБРАЖЕНИЯМИ")
    print("=" * 60)
    
    # Загружаем реальные изображения
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scene_path = os.path.join(base_dir, "visual_search_mvp", "scene.jpg")
    query_path = os.path.join(base_dir, "visual_search_mvp", "red_query.jpg")
    
    if not os.path.exists(scene_path) or not os.path.exists(query_path):
        print("Реальные изображения не найдены!")
        return
    
    scene = cv2.imread(scene_path)
    query = cv2.imread(query_path)
    
    if scene is None or query is None:
        print("Ошибка загрузки изображений!")
        return
    
    print(f"Загружена сцена: {scene.shape}")
    print(f"Загружен запрос: {query.shape}")
    
    # Инициализация счетчика
    counter = WorkingCounter(color_tolerance=120, min_similarity=0.1)
    
    # Поиск объектов
    results = counter.count_objects_working(scene, query)
    
    # Визуализация
    output = scene.copy()
    for obj in results['objects']:
        x, y, w, h = obj['bbox']
        similarity = obj['similarity']
        
        color = (0, 255, 0) if similarity > 0.3 else (0, 255, 255)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 3)
        
        label = f"{similarity:.2f}"
        cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Добавляем query
    h_query, w_query = query.shape[:2]
    scale = min(200 / w_query, 200 / h_query)
    new_w = int(w_query * scale)
    new_h = int(h_query * scale)
    query_resized = cv2.resize(query, (new_w, new_h))
    
    output[10:10+new_h, 10:10+new_w] = query_resized
    cv2.rectangle(output, (10, 10), (10+new_w, 10+new_h), (0, 255, 0), 2)
    
    cv2.imwrite("real_results.jpg", output)
    
    print(f"\nРЕЗУЛЬТАТЫ:")
    print(f"Найдено объектов: {results['total_found']}")
    
    for i, obj in enumerate(results['objects']):
        bbox = obj['bbox']
        similarity = obj['similarity']
        print(f"  Объект {i+1}: similarity={similarity:.3f}")
    
    # Показываем результат
    scale_display = 0.6
    h, w = output.shape[:2]
    output_resized = cv2.resize(output, (int(w * scale_display), int(h * scale_display)))
    
    cv2.imshow("Реальные изображения - результаты", output_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("ЗАПУСК ФИНАЛЬНОГО ТЕСТА")
    print("Эта система должна найти красные объекты на сцене!")
    
    # Сначала идеальный тест
    test_perfect_case()
    
    # Затем реальные изображения
    test_with_real_query()
    
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 60)