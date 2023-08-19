# Проект Кредитного Риск-Менеджмента

В этом проекте мы разработали систему, которая помогает банкам и финансовым учреждениям прогнозировать вероятность невозврата кредита на основе исторических данных и различных характеристик клиента.

## Структура проекта

- **/data:** Директория, содержащая исходные данные для обучения и тестирования модели.
  
- **/src:** Исходный код проекта. Здесь находятся все Python-скрипты, отвечающие за обработку данных, обучение модели и серверную логику.
  
- **/model:** Директория для сохранения и загрузки обученных моделей. 
  
- **/config:** Здесь содержатся конфигурационные файлы, в которых определены различные параметры, такие как путь к данным, параметры модели и т.д.

- **/notebooks:** Директория с Jupyter ноутбуком. Используются для исследовательского анализа данных, прототипирования моделей и визуализации результатов.

- **/img:** Директория со скриншотами и другими изображениями, связанными с проектом. Например, скриншоты рабочего приложения, диаграммы и т. д.

## Как использовать

1. Убедитесь, что все необходимые библиотеки установлены.
2. Запустите сервер, используя команду: `uvicorn server:app --reload`
3. Отправьте POST-запрос на `http://127.0.0.1:8000/predict` со своими данными для получения прогноза.