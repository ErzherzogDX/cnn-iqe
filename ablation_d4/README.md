# Восстановление изображений после сжатия HEVC

Двухмасштабная свёрточная сеть обрабатывает кадр в восьми ориентациях. Каждый прогноз возвращается в исходную ориентацию, затем восемь прогнозов усредняются попиксельно. На 100 тестовых кадрах BSD500 средний RGB-PSNR составил 24,8677 дБ. Выигрыш относительно QECNN составил 0,1301 дБ, относительно декодера x265 — 0,3508 дБ.

## Запуск на готовых весах

Папка `ablation_d4/` располагается в корне копии исходного проекта. Для оценки нужны `testrawyuv/BSD500val.yuv` и `testcompyuv/BSD500valc.yuv`. Для сравнения с QECNN нужен также файл `best_model2026.pth` в корне. Все три файла входят в [исходный репозиторий лабораторной работы](https://ctlab.itmo.ru/gitlab/eabelyaev/cnnimagequalityenhancement/-/tree/af78856948403c3cce7705eb2d7f52f6f44a6824).

Из корня репозитория выполните команды ниже.

```sh
cd ablation_d4
python -m pip install -r requirements.txt
python restore.py evaluate --root .. --out reproduced
python restore.py compare --root .. --single-weights single.pth --d4-weights phase_restorer.pth --reference-csv results/per_image.csv --out repeated
```

Команда `evaluate` сравнивает x265, QECNN и предложенную сеть. В `reproduced/` она сохраняет покадровые значения PSNR, сводку, график и примеры изображений. Команда `compare` применяет обе обученные модели к тем же кадрам и сохраняет в `repeated/` оценки для каждого из четырёх режимов.

## Сколько дало усреднение

| Ориентаций при обучении | Ориентаций при оценке | Средний RGB-PSNR, дБ |
|---:|---:|---:|
| 1 | 1 | 24,7772 |
| 1 | 8 | 24,8148 |
| 8 | 1 | 23,9758 |
| 8 | 8 | 24,8677 |

На фиксированных весах контрольной сети включение усреднения только при оценке дало **+0,0376 дБ**. Если обучать по усреднённому прогнозу и использовать усреднение при оценке, выигрыш относительно сети с одним проходом составил **+0,0905 дБ**. Обе сети обучены с нуля при одинаковой инициализации, на одних фрагментах и с одинаковым числом шагов. Разности рассчитаны до округления.

## Повторное обучение

Для обучения используются пары `trainrawyuv/BSD500train.yuv` и `traincompyuv/BSD500trainc.yuv` из исходного проекта. Для сохранения новых весов выполните из `ablation_d4/` следующие команды.

```sh
python restore.py train --root .. --orientations 1 --weights training/single.pth --training-dir training/single
python restore.py train --root .. --orientations 8 --weights training/d4.pth --training-dir training/d4
```

## Файлы

- `restore.py` и `support.py` содержат всю реализацию, `requirements.txt` перечисляет зависимости.
- `phase_restorer.pth` содержит веса сети, обученной с усреднением, `single.pth` — веса однопроходной контрольной сети.
- [Покадровое сравнение с x265 и QECNN](results/per_image.csv) и [сводка](results/summary.json) содержат результаты основной проверки.
- [Четыре режима усреднения](comparison.csv) и [сводка с SHA-256 весов](summary.json) содержат результаты контрольного эксперимента. Проверки при обучении записаны в `training_single.csv` и `training_d4.csv`.
- [Отчёт в PDF](report/report.pdf), [исходник Typst](report/main.typ) и папка `report/figures/` с рисунками составляют отчёт. Для пересборки из `ablation_d4/` выполните `typst compile report/main.typ report/report.pdf`.
