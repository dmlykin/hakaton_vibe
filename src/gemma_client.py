import base64
import json
import os, os.path
from typing import Dict, Any, Optional
from openai import OpenAI, APIError, APIConnectionError, RateLimitError


class GemmaClient:
    """
    Клиент для работы с моделью Gemma 2 7B через OpenAI-совместимый API.
    Поддерживает выполнение промтов с изображениями и без.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        system_prompt_path: str = "src/system_prompt.txt"
    ):
        """
        Инициализация клиента.

        Args:
            api_key: API ключ для аутентификации
            base_url: Базовый URL API
            system_prompt_path: Путь к файлу с системным промтом
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.system_prompt = self._load_system_prompt(system_prompt_path)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _load_system_prompt(self, path: str) -> str:
        """
        Загружает системный промт из файла.

        Args:
            path: Путь к файлу с системным промтом

        Returns:
            Содержимое системного промта

        Raises:
            FileNotFoundError: Если файл промта не найден
            IOError: Если возникла ошибка при чтении файла
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл системного промта не найден: {path}")
        except Exception as e:
            raise IOError(f"Ошибка при чтении системного промта: {e}")

    def _encode_image(self, image_path: str) -> str:
        """
        Кодирует изображение в формат base64.

        Args:
            image_path: Путь к изображению

        Returns:
            Строка с base64-кодированным изображением

        Raises:
            FileNotFoundError: Если файл изображения не найден
            IOError: Если возникла ошибка при чтении изображения
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")
        except Exception as e:
            raise IOError(f"Ошибка при кодировании изображения: {e}")

    def execute_with_image(
        self,
        prompt: str,
        image_path: str,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Выполняет промт с изображением.

        Args:
            prompt: Пользовательский промт
            image_path: Путь к изображению
            temperature: Параметр температуры для генерации
            max_tokens: Максимальное количество токенов в ответе

        Returns:
            Строка с ответом модели в формате JSON

        Raises:
            ValueError: При ошибках валидации входных данных
            httpx.RequestError: При ошибках сети
            Exception: При других ошибках
        """
        if not prompt.strip():
            raise ValueError("Промт не может быть пустым")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл изображения не существует: {image_path}")

        # Кодируем изображение
        base64_image = self._encode_image(image_path)

        # Формируем сообщения
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        # Формируем payload
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": ["</json>", "```"]
        }

        # Отправляем запрос
        try:
            response = self.client.chat.completions.create(**payload)
            
            if len(response.choices) == 0:
                raise Exception("Ответ модели не содержит поле choices")
                
            content = response.choices[0].message.content.strip()
            return self._clean_response(content)

        except APIConnectionError as e:
            raise Exception(f"Ошибка подключения к API: {e}")
        except RateLimitError as e:
            raise Exception(f"Превышен лимит запросов к API: {e}")
        except APIError as e:
            raise Exception(f"API ошибка: {e}")
        except Exception as e:
            raise Exception(f"Ошибка при выполнении запроса с изображением: {e}")

    def execute_without_image(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Выполняет текстовый промт без изображения.

        Args:
            prompt: Пользовательский промт
            temperature: Параметр температуры для генерации
            max_tokens: Максимальное количество токенов в ответе

        Returns:
            Строка с ответом модели в формате JSON

        Raises:
            ValueError: При ошибках валидации входных данных
            httpx.RequestError: При ошибках сети
            Exception: При других ошибках
        """
        if not prompt.strip():
            raise ValueError("Промт не может быть пустым")

        # Формируем сообщения
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Формируем payload
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": ["</json>", "```"]
        }

        # Отправляем запрос
        try:
            response = self.client.chat.completions.create(**payload)
            
            if len(response.choices) == 0:
                raise Exception("Ответ модели не содержит поле choices")
                
            content = response.choices[0].message.content.strip()
            return self._clean_response(content)

        except APIConnectionError as e:
            raise Exception(f"Ошибка подключения к API: {e}")
        except RateLimitError as e:
            raise Exception(f"Превышен лимит запросов к API: {e}")
        except APIError as e:
            raise Exception(f"API ошибка: {e}")
        except Exception as e:
            raise Exception(f"Ошибка при выполнении текстового запроса: {e}")

    def _clean_response(self, content: str) -> str:
        """
        Очищает ответ модели от лишних символов и форматирует как чистый JSON.

        Args:
            content: Сырой ответ модели

        Returns:
            Очищенный JSON-ответ
        """
        # Удаляем Markdown-разметку
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        # Пытаемся распарсить и вернуть валидный JSON
        try:
            parsed = json.loads(content)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            # Если не получилось распарсить, возвращаем как есть
            return content
