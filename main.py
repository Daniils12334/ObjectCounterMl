import cv2
import argparse
import logging
from src.core.segmenters import Segmenter
from src.core.embedders import Embedder
from src.core.search_engine import VisualSearchEngine
from src.utils.visualization import draw_results

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Поиск похожих объектов на сцене')
    parser.add_argument('--scene', type=str, required=True, help='Путь к изображению сцены')
    parser.add_argument('--query', type=str, required=True, help='Путь к эталонному изображению')
    parser.add_argument('--output', type=str, default='result.jpg', help='Путь для сохранения результата')
    parser.add_argument('--threshold', type=float, default=0.75, help='Порог сходства')
    
    args = parser.parse_args()
    
    try:
        # Загрузка изображений
        logger.info("Загрузка изображений...")
        scene_image = cv2.imread(args.scene)
        query_image = cv2.imread(args.query)
        
        if scene_image is None:
            raise ValueError(f"Не удалось загрузить изображение сцены: {args.scene}")
        if query_image is None:
            raise ValueError(f"Не удалось загрузить эталонное изображение: {args.query}")
        
        # Инициализация компонентов
        logger.info("Инициализация моделей...")
        segmenter = Segmenter()
        embedder = Embedder()
        search_engine = VisualSearchEngine(segmenter, embedder, args.threshold)
        
        # Поиск похожих объектов
        logger.info("Запуск поиска...")
        results = search_engine.search_similar_objects(scene_image, query_image)
        
        # Визуализация результатов
        logger.info("Визуализация результатов...")
        output_image = draw_results(scene_image, results['matches'], query_image)
        
        # Сохранение результата
        cv2.imwrite(args.output, output_image)
        
        # Вывод статистики
        print(f"\n=== РЕЗУЛЬТАТЫ ПОИСКА ===")
        print(f"Всего объектов на сцене: {results['total_objects']}")
        print(f"Найдено похожих объектов: {len(results['matches'])}")
        print(f"Порог сходства: {args.threshold}")
        
        for i, match in enumerate(results['matches']):
            print(f"Объект {i+1}: сходство = {match['similarity']:.3f}")
        
        print(f"\nРезультат сохранен в: {args.output}")
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        raise

if __name__ == "__main__":
    main()