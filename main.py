import os
import torch
from pathlib import Path
from ultralytics import YOLO
from flask import Flask, render_template_string, request, jsonify, send_from_directory
import base64
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from collections import defaultdict
import uuid

app = Flask(__name__)
app.secret_key = 'traffic_detector_secret_key_2024'


class EnsembleDetector:
    def __init__(self):
        self.models = []
        self.model_names = []
        self.classes = ['pedestrian', 'car', 'motorbike', 'truck']
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        self.device = '0' if torch.cuda.is_available() else 'cpu'

        # 1. Ваша модель
        self.custom_model_path = 'my_training/custom_model/weights/best.pt'
        self.custom_model = None

        # 2. Предобученная модель YOLOv8
        self.pretrained_models = {
            'yolov8n': 'yolov8n.pt',
        }

        # Маппинг COCO классов на наши
        self.coco_to_our = {
            0: 0,  # person -> pedestrian
            1: -1,  # bicycle -> игнорируем
            2: 1,  # car -> car
            3: 2,  # motorcycle -> motorbike
            5: 1,  # bus -> car
            7: 3,  # truck -> truck
        }

        # Веса моделей
        self.model_weights = {
            'custom': 1.0,
            'yolov8n': 3.0,  # Больше вес для YOLO
        }

        # Пороговые значения
        self.containment_threshold = 0.6  # 60% внутри

        # Русские названия для отображения
        self.russian_names = {
            'pedestrian': 'Пешеход',
            'car': 'Машина',
            'motorbike': 'Мотоцикл',
            'truck': 'Грузовик'
        }

        self.load_all_models()

    def load_all_models(self):
        """Загружаем все доступные модели"""
        print("\n" + "=" * 60)
        print("🔄 ЗАГРУЗКА АНСАМБЛЯ МОДЕЛЕЙ")
        print("=" * 60)

        # 1. Пытаемся загрузить вашу модель
        if os.path.exists(self.custom_model_path):
            try:
                print(f"📥 Загружаем ВАШУ модель: {self.custom_model_path}")
                self.custom_model = YOLO(self.custom_model_path)
                self.models.append(('custom', self.custom_model))
                print(f"✅ Ваша модель загружена")
            except Exception as e:
                print(f"❌ Ошибка загрузки вашей модели: {e}")
        else:
            print("⚠️  Ваша модель не найдена")

        # 2. Загружаем предобученная модель
        for name, path in self.pretrained_models.items():
            try:
                print(f"📥 Загружаем {name}...")
                model = YOLO(path)
                self.models.append((name, model))
                print(f"✅ {name} загружена")
            except Exception as e:
                print(f"❌ Ошибка загрузки {name}: {e}")

        print(f"🎯 Всего загружено моделей: {len(self.models)}")
        print("=" * 60)

        return len(self.models) > 0

    def map_coco_to_our_classes(self, class_id, class_name):
        """Маппим COCO классы на наши"""
        if class_id in self.coco_to_our:
            our_id = self.coco_to_our[class_id]
            if our_id == -1:
                return None, -1
            return self.classes[our_id], our_id

        class_name_lower = class_name.lower()
        if 'person' in class_name_lower:
            return 'pedestrian', 0
        elif 'motor' in class_name_lower:
            return 'motorbike', 2
        elif 'car' in class_name_lower or 'bus' in class_name_lower:
            return 'car', 1
        elif 'truck' in class_name_lower:
            return 'truck', 3

        return None, -1

    def ensemble_predict(self, image, conf_threshold=0.25):
        """Предсказание с помощью ансамбля моделей"""
        all_detections = []

        for model_name, model in self.models:
            try:
                results = model.predict(
                    source=image,
                    conf=conf_threshold,
                    iou=0.45,
                    imgsz=640,
                    verbose=False,
                    augment=False
                )

                if results and len(results) > 0:
                    result = results[0]

                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes

                        for box in boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                            if hasattr(model, 'names') and model.names:
                                detected_name = model.names[cls_id]
                            else:
                                detected_name = f"class_{cls_id}"

                            # Маппинг классов
                            our_class = None
                            our_id = -1

                            if model_name != 'custom':
                                our_class, our_id = self.map_coco_to_our_classes(cls_id, detected_name)
                            else:
                                # Для вашей модели
                                detected_lower = detected_name.lower()
                                if 'person' in detected_lower:
                                    our_class, our_id = 'pedestrian', 0
                                elif 'car' in detected_lower:
                                    our_class, our_id = 'car', 1
                                elif 'motor' in detected_lower or 'bike' in detected_lower:
                                    our_class, our_id = 'motorbike', 2
                                elif 'truck' in detected_lower:
                                    our_class, our_id = 'truck', 3

                            if our_class is None or our_id == -1:
                                continue

                            weight = self.model_weights.get(model_name, 1.0)
                            weighted_conf = conf * weight

                            all_detections.append({
                                'model': model_name,
                                'class': our_class,
                                'class_id': our_id,
                                'confidence': conf,
                                'weighted_confidence': weighted_conf,
                                'box': [float(x1), float(y1), float(x2), float(y2)],
                                'area': (x2 - x1) * (y2 - y1)
                            })

            except Exception as e:
                print(f"⚠️  Ошибка предсказания {model_name}: {e}")
                continue

        return all_detections

    def calculate_iou(self, box1, box2):
        """Вычисляем Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return max(0.0, iou)

    def calculate_containment(self, box1, box2):
        """Вычисляет долю box2 внутри box1"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        if box2_area == 0:
            return 0.0

        return intersection_area / box2_area

    def fuse_detections(self, detections):
        """Объединяем детекции с исправлением мотоциклов и грузовиков"""
        if not detections:
            return []

        print(f"\n🔄 Объединение {len(detections)} детекций")

        # 1. Находим все мотоциклы от YOLO и грузовики от YOLO
        yolo_motorbikes = [d for d in detections if d['model'] == 'yolov8n' and d['class'] == 'motorbike']
        yolo_trucks = [d for d in detections if d['model'] == 'yolov8n' and d['class'] == 'truck']

        print(f"   🏍️  YOLO мотоциклы: {len(yolo_motorbikes)}")
        print(f"   🚚 YOLO грузовики: {len(yolo_trucks)}")

        # 2. Удаляем машины внутри мотоциклов И грузовиков
        detections_to_remove = []

        # Для мотоциклов
        for mb_idx, motorbike in enumerate(yolo_motorbikes):
            motorbike_box = motorbike['box']

            for i, det in enumerate(detections):
                if det['class'] == 'car' and det['model'] == 'custom':
                    car_box = det['box']
                    containment = self.calculate_containment(motorbike_box, car_box)

                    if containment > self.containment_threshold:
                        print(f"   🚫 Удаляем машину внутри мотоцикла #{mb_idx + 1}")
                        detections_to_remove.append(i)

        # Для грузовиков
        for truck_idx, truck in enumerate(yolo_trucks):
            truck_box = truck['box']

            for i, det in enumerate(detections):
                if det['class'] == 'car' and det['model'] == 'custom':
                    car_box = det['box']
                    containment = self.calculate_containment(truck_box, car_box)

                    if containment > self.containment_threshold:
                        print(f"   🚫 Удаляем машину внутри грузовика #{truck_idx + 1}")
                        detections_to_remove.append(i)

        # Удаляем дубликаты
        detections_to_remove = list(set(detections_to_remove))

        # 3. Фильтруем детекции
        filtered_detections = [d for i, d in enumerate(detections) if i not in detections_to_remove]
        print(f"   📉 Удалено детекций: {len(detections_to_remove)}")

        # 4. Объединяем оставшиеся детекции
        filtered_detections.sort(key=lambda x: x['weighted_confidence'], reverse=True)
        fused = []
        used = [False] * len(filtered_detections)

        for i in range(len(filtered_detections)):
            if used[i]:
                continue

            current = filtered_detections[i]
            current_box = current['box']
            current_class = current['class']
            current_model = current['model']

            similar_detections = [current]

            for j in range(i + 1, len(filtered_detections)):
                if used[j]:
                    continue

                other = filtered_detections[j]
                other_box = other['box']
                other_class = other['class']
                other_model = other['model']

                iou = self.calculate_iou(current_box, other_box)

                # Правило для мотоциклов: YOLO имеет приоритет
                if iou > 0.3:
                    if ((current_model == 'yolov8n' and current_class == 'motorbike' and
                         other_model == 'custom' and other_class == 'car') or
                            (other_model == 'yolov8n' and other_class == 'motorbike' and
                             current_model == 'custom' and current_class == 'car')):

                        # YOLO мотоцикл побеждает
                        if current_model == 'yolov8n' and current_class == 'motorbike':
                            used[j] = True
                            continue
                        else:
                            used[i] = True
                            break

                    # Правило для грузовиков: YOLO имеет приоритет
                    elif ((current_model == 'yolov8n' and current_class == 'truck' and
                           other_model == 'custom' and other_class == 'car') or
                          (other_model == 'yolov8n' and other_class == 'truck' and
                           current_model == 'custom' and current_class == 'car')):

                        # YOLO грузовик побеждает
                        if current_model == 'yolov8n' and current_class == 'truck':
                            used[j] = True
                            continue
                        else:
                            used[i] = True
                            break

                    # Объединяем одинаковые классы
                    elif current_class == other_class and iou > 0.3:
                        similar_detections.append(other)
                        used[j] = True

            if used[i]:
                continue

            # Объединяем группу
            if len(similar_detections) > 1:
                fused_det = self.merge_similar_detections(similar_detections)
            else:
                fused_det = current

            fused.append(fused_det)
            used[i] = True

        print(f"   ✅ Финальных детекций: {len(fused)}")
        return fused

    def merge_similar_detections(self, detections):
        """Объединяем похожие детекции"""
        total_weight = 0
        weighted_box = [0, 0, 0, 0]
        weighted_conf = 0

        for det in detections:
            weight = det['weighted_confidence']
            total_weight += weight

            box = det['box']
            for i in range(4):
                weighted_box[i] += box[i] * weight

            weighted_conf += det['confidence'] * weight

        if total_weight > 0:
            box = [coord / total_weight for coord in weighted_box]
            confidence = weighted_conf / total_weight
        else:
            box = detections[0]['box']
            confidence = detections[0]['confidence']

        best_class = detections[0]['class']
        class_id = detections[0]['class_id']

        return {
            'class': best_class,
            'class_id': class_id,
            'confidence': confidence,
            'box': box,
            'models': [d['model'] for d in detections],
            'count': len(detections)
        }

    def predict_image(self, image_path):
        """Основная функция предсказания"""
        if not self.models:
            print("❌ Нет загруженных моделей!")
            return None, None, None

        print(f"\n🔍 Анализируем: {image_path}")

        try:
            # Открываем изображение с помощью PIL для поддержки всех форматов
            pil_img = Image.open(image_path)

            # Конвертируем RGBA в RGB если нужно
            if pil_img.mode in ('RGBA', 'LA', 'P'):
                # Создаем белый фон для прозрачных изображений
                background = Image.new('RGB', pil_img.size, (255, 255, 255))
                if pil_img.mode == 'P':
                    pil_img = pil_img.convert('RGBA')
                background.paste(pil_img, mask=pil_img.split()[-1] if pil_img.mode == 'RGBA' else None)
                pil_img = background
            elif pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

            # Конвертируем PIL в OpenCV формат для детекции
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            height, width = img.shape[:2]

            print(f"📏 Размер изображения: {width}x{height}")

            all_detections = self.ensemble_predict(img, conf_threshold=0.2)

            fused_detections = self.fuse_detections(all_detections)

            stats = {
                'total': len(fused_detections),
                'pedestrian': 0, 'car': 0, 'motorbike': 0, 'truck': 0,
                'objects': [],
                'model_stats': defaultdict(int)
            }

            print(f"\n🎯 Финальные результаты:")

            # Создаем PIL изображение для отрисовки (чтобы избежать проблем с шрифтами)
            result_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(result_img)

            # Пытаемся загрузить нормальный шрифт, иначе используем стандартный
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
                except:
                    font = ImageFont.load_default()

            for i, det in enumerate(fused_detections):
                class_name = det['class']
                confidence = det['confidence']
                box = det['box']
                models_used = det.get('models', [])

                if class_name in self.classes:
                    stats[class_name] += 1

                for model_name in models_used:
                    stats['model_stats'][model_name] += 1

                # Получаем русское название
                russian_name = self.russian_names.get(class_name, class_name)

                stats['objects'].append({
                    'id': i + 1,
                    'name': russian_name,
                    'english_name': class_name,
                    'confidence': confidence,
                    'confidence_percent': f"{confidence:.1%}",
                    'box': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    'models': ', '.join(models_used) if models_used else 'ensemble'
                })

                print(f"   {i + 1}. {russian_name} ({class_name}) - {confidence:.1%}")

                # Определяем цвет для класса
                if class_name in self.classes:
                    color_idx = self.classes.index(class_name)
                else:
                    color_idx = 0

                color = self.colors[color_idx % len(self.colors)]
                x1, y1, x2, y2 = [int(coord) for coord in box]

                # Рисуем прямоугольник на PIL изображении
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                # Подготовка текста с русским названием
                label = f"{russian_name} {confidence:.1%}"

                # Получаем размер текста
                try:
                    text_bbox = draw.textbbox((x1, y1 - 20), label, font=font)
                except:
                    # Если шрифт не поддерживает кириллицу, используем английское название
                    label = f"{class_name} {confidence:.1%}"
                    text_bbox = draw.textbbox((x1, y1 - 20), label, font=font)

                # Рисуем фон для текста
                draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
                               fill=color)

                # Рисуем текст
                draw.text((x1, y1 - 20), label, font=font, fill=(255, 255, 255))

            print(
                f"\n📈 Сводка: Пешеходы={stats['pedestrian']}, Машины={stats['car']}, Мотоциклы={stats['motorbike']}, Грузовики={stats['truck']}")

            # Добавляем информационную панель
            info_text = f"Объектов: {stats['total']}"

            try:
                text_bbox = draw.textbbox((10, 10), info_text, font=font)
            except:
                # Если не поддерживает кириллицу
                info_text = f"Objects: {stats['total']}"
                text_bbox = draw.textbbox((10, 10), info_text, font=font)

            # Рисуем фон для информационной панели
            draw.rectangle([text_bbox[0] - 5, text_bbox[1] - 5, text_bbox[2] + 5, text_bbox[3] + 5],
                           fill=(0, 0, 0))

            # Рисуем текст информационной панели
            draw.text((10, 10), info_text, font=font, fill=(0, 255, 0))

            # Конвертируем обратно в OpenCV формат для base64 кодирования
            result_img_cv = cv2.cvtColor(np.array(result_img), cv2.COLOR_RGB2BGR)

            # Кодируем изображение в base64
            _, buffer = cv2.imencode('.jpg', result_img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            return stats, fused_detections, img_base64

        except Exception as e:
            print(f"❌ Ошибка предсказания: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None


# Создаем детектор
detector = EnsembleDetector()


# Функции для загрузки HTML файлов напрямую
def load_html_file(filename):
    """Загружает HTML файл из корневой директории"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return None


@app.route('/')
def index():
    html_content = load_html_file('index.html')
    if html_content:
        return render_template_string(html_content)
    else:
        return "Файл index.html не найден в корневой директории", 404


@app.route('/result')
def result():
    html_content = load_html_file('result.html')
    if html_content:
        return render_template_string(html_content)
    else:
        return "Файл result.html не найден в корневой директории", 404


@app.route('/style.css')
def serve_css():
    return send_from_directory('.', 'style.css')


@app.route('/script.js')
def serve_js():
    return send_from_directory('.', 'script.js')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Файл не найден'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Файл не выбран'})

    print(f"\n📥 Получен файл: {file.filename}")

    # Проверка расширения файла
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp',
                          '.JPG', '.JPEG', '.PNG', '.WEBP', '.GIF', '.BMP'}
    file_ext = os.path.splitext(file.filename)[1]
    if file_ext not in allowed_extensions:
        return jsonify({
            'success': False,
            'error': f'Неподдерживаемый формат файла: {file_ext}. Используйте JPG, PNG, WEBP или GIF'
        })

    # Сохраняем временный файл
    temp_filename = f"temp_{uuid.uuid4().hex[:8]}{file_ext}"
    temp_path = temp_filename

    try:
        file.save(temp_path)
        print(f"✅ Файл сохранен: {temp_path} (размер: {os.path.getsize(temp_path)} байт)")

        stats, detections, img_base64 = detector.predict_image(temp_path)

        if stats:
            probabilities = {}
            total = stats['total'] if stats['total'] > 0 else 1

            for class_name in detector.classes:
                count = stats.get(class_name, 0)
                probability = count / total
                probabilities[class_name] = {
                    'count': count,
                    'probability': probability,
                    'probability_percent': f"{probability:.1%}"
                }

            # ОТЛАДОЧНЫЙ ВЫВОД
            print(f"\n📊 СТАТИСТИКА ДЛЯ ОТПРАВКИ:")
            print(f"  Всего объектов: {stats['total']}")
            print(f"  Вероятности: {probabilities}")
            print(f"  Количество детекций в objects: {len(stats.get('objects', []))}")

            if stats.get('objects'):
                for i, obj in enumerate(stats['objects']):
                    print(f"  Объект {i + 1}: {obj}")

            results = {
                'image': f"data:image/jpeg;base64,{img_base64}" if img_base64 else '',
                'total': stats['total'],
                'detections': stats.get('objects', []),
                'probabilities': probabilities
            }

            print(f"✅ Обработка завершена. Обнаружено объектов: {stats['total']}")

            response = jsonify({
                'success': True,
                'results': results,
                'message': f'Обнаружено {stats["total"]} объектов'
            })

            print(f"📤 Отправляем ответ клиенту")
            return response

        else:
            print("⚠️  Объекты не обнаружены")
            results = {
                'image': '',
                'total': 0,
                'detections': [],
                'probabilities': {cls: {'count': 0, 'probability': 0, 'probability_percent': '0%'}
                                  for cls in detector.classes}
            }

            return jsonify({
                'success': True,
                'results': results,
                'message': 'Объекты не обнаружены'
            })

    except Exception as e:
        print(f"❌ Ошибка обработки: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

    finally:
        # Удаляем временный файл
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"🗑️  Временный файл удален: {temp_path}")
            except Exception as e:
                print(f"⚠️  Не удалось удалить временный файл: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚦 Traffic Detector")
    print("=" * 60)
    print("🎯 Классы: Пешеходы, Машины, Мотоциклы, Грузовики")
    print("🤝 Ансамбль: ваша модель + YOLOv8")
    print("📁 Поддерживаемые форматы: JPG, PNG, WEBP, GIF, BMP")
    print("=" * 60)
    print("📌 Откройте: http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
