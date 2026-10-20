# ASR Eval API

[![CI](https://github.com/ArtemiyMosin/asr-eval-api/actions/workflows/ci.yml/badge.svg)](https://github.com/ArtemiyMosin/asr-eval-api/actions/workflows/ci.yml)

Небольшой сервис на FastAPI для оценки качества распознавания речи (ASR).
Он сравнивает текст, который выдала модель распознавания (*hypothesis*), с эталонной расшифровкой
(*reference*) и считает **WER** (word error rate) и **CER** (character error rate). Отдельно показывает
число замен, удалений и вставок. Результаты можно сохранять, фильтровать, смотреть сводку по модели
и сравнивать модели в лидерборде.

**Для кого:** для разработчиков и ML-инженеров, которые выбирают или дообучают модель распознавания
(Whisper, T-one и т.п.) и хотят быстро сравнить модели на своих примерах без ноутбуков и скриптов.

## Возможности

| Метод | Маршрут | Что делает |
|---|---|---|
| GET | `/` | Краткая информация о сервисе |
| GET | `/health` | Проверка работоспособности: статус, uptime, число сохранённых оценок |
| GET | `/version` | Имя и версия приложения |
| POST | `/metrics/wer` | Посчитать WER/CER для пары текстов, ничего не сохраняя |
| POST | `/evaluations` | Оценить результат модели и сохранить его |
| GET | `/evaluations?model=&min_wer=&limit=` | Список оценок с фильтром по модели и минимальному WER (поиск плохих примеров) |
| GET | `/evaluations/{id}` | Одна оценка по идентификатору (404, если её нет) |
| DELETE | `/evaluations/{id}` | Удалить оценку |
| GET | `/models/{model_name}/summary` | Сводка по модели: корпусный WER, средние WER/CER, худший пример |
| GET | `/leaderboard` | Модели по возрастанию корпусного WER |

Перед подсчётом тексты нормализуются: нижний регистр, `ё → е`, знаки препинания удаляются, лишние
пробелы схлопываются. Поэтому `«Привет, мир!»` и `привет мир` считаются совпадением.

Корпусный WER равен сумме ошибок по всем примерам, делённой на суммарное число слов в эталонах.
Он честнее среднего WER по фразам: короткая фраза из одного слова не перевешивает длинную.

Данные хранятся в памяти процесса. После перезапуска они пропадают: для учебного сервиса база не нужна.

## Установка

Нужен Python 3.10+.

```bash
git clone https://github.com/ArtemiyMosin/asr-eval-api.git
cd asr-eval-api

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt        # только для запуска
pip install -r requirements-dev.txt    # для запуска и тестов
```

## Переменные окружения

Все переменные необязательные, у каждой есть значение по умолчанию. Пример лежит в `.env.example`.

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `APP_NAME` | `asr-eval-api` | Имя сервиса в логах и в `/version` |
| `APP_VERSION` | `1.0.0` | Версия, которую отдаёт `/version` |
| `LOG_LEVEL` | `INFO` | Уровень логирования: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `MAX_TEXT_LENGTH` | `1000` | Максимальная длина reference/hypothesis в символах. Более длинные тексты отклоняются с кодом 422 |
| `WER_ALERT_THRESHOLD` | `0.3` | Оценки с WER выше порога пишутся в лог как `WARNING` |

Чтобы взять значения из файла:

```bash
cp .env.example .env               # .env в git не попадает (см. .gitignore)
set -a && source .env && set +a
```

## Запуск

Обычный запуск, без `--reload`:

```bash
uvicorn app_final:app --host 127.0.0.1 --port 8000
```

Для разработки, с перезапуском при изменении кода:

```bash
uvicorn app_final:app --reload
```

Интерактивная документация Swagger доступна по адресу http://127.0.0.1:8000/docs.
Там можно отправить любой запрос прямо из браузера.

### Пример

```bash
curl -X POST http://127.0.0.1:8000/evaluations \
  -H "Content-Type: application/json" \
  -d '{"model": "whisper-small", "audio_id": "call_01.wav",
       "reference": "Включи свет на кухне", "hypothesis": "включи свет в кухне"}'

curl http://127.0.0.1:8000/models/whisper-small/summary
curl http://127.0.0.1:8000/leaderboard
```

## Тесты

```bash
python -m pytest -v
```

Тесты в `test_app.py` покрывают `/health`, `/version`, нормализацию, подсчёт замен, удалений и вставок,
ошибки валидации (пустой эталон, слишком длинный текст, неверное имя модели), 404 для несуществующих
записей, фильтры списка, сводку и лидерборд.

## CI

Workflow `.github/workflows/ci.yml` запускается при каждом `push` и `pull request`:

1. устанавливает зависимости из `requirements-dev.txt`;
2. запускает `python -m pytest -v`;
3. поднимает сервис через `uvicorn` и проверяет `/health`, `/version` и `/metrics/wer` через `curl`;
4. после успешных проверок загружает artifact `asr-eval-api-<commit>` с файлами приложения
   (`app_final.py`, `requirements.txt`, `.env.example`, `README.md`).

Artifact можно скачать на странице запуска во вкладке **Actions**, раздел *Artifacts*.

## Диагностика

**Сервис жив?**

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","uptime_seconds":12.3,"stored_evaluations":0}
```

**Какая версия запущена?**

```bash
curl http://127.0.0.1:8000/version
# {"name":"asr-eval-api","version":"1.0.0"}
```

**Где смотреть логи.** Логи пишутся в stdout, в тот же терминал, где запущен `uvicorn`. Чтобы сохранить их в файл:

```bash
uvicorn app_final:app --host 127.0.0.1 --port 8000 2>&1 | tee app.log
```

Что пишется в лог:

- при старте: версия и действующие настройки (`Starting asr-eval-api v1.0.0 (log level INFO, ...)`);
- каждый запрос: метод, путь, статус и время (`POST /evaluations -> 201 (2.2 ms)`);
- каждая сохранённая оценка: модель, файл, WER и CER. Если WER выше `WER_ALERT_THRESHOLD`,
  запись идёт с уровнем `WARNING`;
- остановка сервиса.

Для подробных логов запустите с `LOG_LEVEL=DEBUG`.

**Частые проблемы**

| Симптом | Причина и решение |
|---|---|
| `Address already in use` | Порт 8000 занят. Запустите с `--port 8001` или остановите другой процесс (`lsof -i :8000`) |
| `ModuleNotFoundError: fastapi` | Не активировано окружение или не установлены зависимости: `source .venv/bin/activate && pip install -r requirements.txt` |
| `422` на `/metrics/wer` | Эталон пустой после нормализации (одна пунктуация), текст длиннее `MAX_TEXT_LENGTH` или в имени модели есть пробелы и спецсимволы |
| `404` на `/models/<имя>/summary` | Для этой модели ещё нет ни одной оценки |
| Данные пропали | Хранилище в памяти и очищается при перезапуске. Так задумано |

## Структура

```
.
├── app_final.py              # приложение
├── test_app.py               # тесты
├── requirements.txt          # зависимости для запуска
├── requirements-dev.txt      # + зависимости для тестов
├── .env.example              # пример переменных окружения
├── .gitignore
└── .github/workflows/ci.yml  # GitHub Actions
```
