# НКС ЛР №3 — Развертывание дообученной языковой модели

## Описание

В проекте реализовано развертывание дообученной языковой модели для лабораторной работы №3 по дисциплине «Нейрокомпьютерные системы».

Используется базовая модель `Qwen/Qwen3-1.7B` и LoRA-адаптер `qwen3_book_lora_adapter`, обученный в ЛР №2 на корпусе по книге *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*.

Реализованы два режима работы:

1. Консольный чат: `run_chat.py`
2. Web UI + FastAPI: `api_server.py` и `static/index.html`

## Структура проекта

```text
run_chat.py
api_server.py
requirements.txt
static/index.html
screenshots/
qwen3_book_lora_adapter/