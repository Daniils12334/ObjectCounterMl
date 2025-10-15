from core.search_engine import ImprovedVisualSearchEngine
from core.utils import ImageUtils, ColorAnalyzer
import cv2
import numpy as np

def create_color_test_images():
    """Создаём тестовые изображения с разными цветами"""
    # Сцена с разноцветными объектами
    scene = np.ones((400, 600, 3), dtype=np.uint8) * 240  # Светло-серый фон
    
    # Чётко разные цвета в BGR
    colors = {
        'red': (0, 0, 255),      # Ярко-красный
        'blue': (255, 0, 0),     # Ярко-синий  
        'green': (0, 255, 0),    # Ярко-зелёный
        'yellow': (0, 255, 255), # Ярко-жёлтый
    }
    
    positions = [(50, 50), (200, 50), (350, 50), (500, 50)]
    
    for (x, y), color_name in zip(positions, colors.keys()):
        color_bgr = colors[color_name]
        cv2.rectangle(scene, (x, y), (x+80, y+80), color_bgr, -1)
        cv2.putText(scene, color_name, (x, y+100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    cv2.imwrite("scene2.jpg", scene)
    
    # Query - красный квадрат
    query = np.ones((90, 90, 3), dtype=np.uint8) * 240
    cv2.rectangle(query, (5, 5), (85, 85), (0, 0, 255), -1)  # Красный
    cv2.imwrite("query2.jpg", query)
    
    print("✅ Created color test images")

def test_color_search():
    """Тестируем поиск с цветовой фильтрацией"""
    print("🎨 Testing Color-Aware Search...")
    
    # Создаём тестовые изображения
    create_color_test_images()
    
    # Инициализируем улучшенный движок
    engine = ImprovedVisualSearchEngine()
    
    # Индексируем сцену
    print("📸 Indexing color test scene...")
    engine.index_image("scene2.jpg", "color_test")
    
    # ТЕСТ 1: Обычный поиск (найдёт все квадраты)
    print("\n🔍 TEST 1: Normal search (finds all squares)...")
    normal_results = engine.search("query2.jpg", top_k=4, threshold=0.3)
    print(f"Normal search found {len(normal_results)} objects")
    
    # ТЕСТ 2: Поиск с фильтрацией по цвету
    print("\n🔍 TEST 2: Color-filtered search (should find only RED)...")
    color_results = engine.search_with_color_filter(
        "red_query.jpg", 
        target_color="red",
        color_tolerance=25,
        top_k=4, 
        threshold=0.3
    )
    print(f"Color-filtered search found {len(color_results)} RED objects")
    
    # ТЕСТ 3: Комбинированный поиск
    print("\n🔍 TEST 3: Combined shape + color search...")
    combined_results = engine.search_combined(
        "red_query.jpg",
        target_color="red", 
        shape_weight=0.6,  # 60% важность формы
        color_weight=0.4,  # 40% важность цвета
        top_k=4
    )
    print(f"Combined search found {len(combined_results)} objects")
    
    # Визуализируем все результаты
    if normal_results:
        print("\n🖼️ Visualizing NORMAL search results...")
        ImageUtils.visualize_search_results("query2.jpg", normal_results)
    
    if color_results:
        print("\n🖼️ Visualizing COLOR-FILTERED results...")
        ImageUtils.visualize_search_results("query2.jpg", color_results)

if __name__ == "__main__":
    test_color_search()