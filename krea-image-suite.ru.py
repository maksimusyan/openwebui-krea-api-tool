"""
title: Krea Image Suite
author: Open WebUI Admin
author_url: https://api.krea.ai
description: Генерация и обработка изображений через Krea API (text→image, редактирование, апскейл). Модели: Seedream, Qwen, Krea 2, Nano Banana, GPT Image, Flux, SeedEdit, Grok, Ideogram, Krea Enhance, Topaz и др. Умеет сам доставать байты картинок из чата Open WebUI и из файлового хранилища Open Terminal, следит за памятью, лимитами и не теряет job_id.
required_open_webui_version: 0.4.0
requirements: requests
version: 2.3.0
licence: MIT
"""

# =============================================================================
# Krea Image Suite — tool для Open WebUI.
#
# ЧТО ИЗМЕНЕНО В 2.1 (кратко, подробности в krea-plan.md):
#  1. Все блокирующие HTTP-вызовы ушли в отдельный поток (asyncio.to_thread):
#     раньше синхронный requests внутри async-метода замораживал event loop
#     Open WebUI на всё время ожидания (до 180–240 c) — это и давало «таймауты».
#  2. job_id больше не теряется: submit защищён try/except, при обрыве связи
#     задача разыскивается через GET /jobs (окно по created_at), а все отправленные
#     задачи пишутся в локальный реестр (последние 50) с алиасами last/last:2.
#  3. Есть проверка сервиса krea_status (ключ, сеть, задержка, активные задачи,
#     последние задачи) и find_jobs для восстановления потерянных job_id.
#  4. Защита памяти: потоковое скачивание с жёстким лимитом байт, LRU-кэш
#     указателей, ограничение размера JSON-тела, а слишком большие картинки не
#     вставляются в payload как data URI (maxLength=1024 у media-полей НЕ
#     проверяется — проверка A 20.09.2026; ограничение теперь по размеру тела) —
#     они автоматически заливаются в POST /assets, а в запрос идёт короткий URL.
#  5. Таблица моделей MODEL_SPEC генерируется из OpenAPI-спеки
#     (krea-work/build_model_spec.py) — поля, обязательные поля, enum'ы
#     aspect_ratio/resolution, лимиты референсов и цены больше не «на глаз».
#  6. Обработка 401/402/400/404/429/5xx с человеческими подсказками,
#     ретраи с бэкоффом, прогрессивный интервал опроса, жёсткий потолок wait_s,
#     защита от превышения лимита одновременных задач (429 = «maximum number
#     of concurrent jobs»).
#
# ЧТО ИЗМЕНЕНО В 2.2 (подробности в krea-plan.md, сессия S3):
#  1. upload_asset берёт размеры и вес картинки ИЗ ОТВЕТА Krea (POST /assets отдаёт
#     width/height/size_bytes/mime_type — проверка B 20.09.2026), локальный разбор
#     заголовка файла остался только резервом. Тот же ответ заполняет кэш размеров
#     (_size_by_pointer), откуда апскейл узнаёт разрешение исходника.
#  2. Регрессионный тест «upload_asset не блокирует event loop» (дефект P0-E)
#     перенесён из отдельного скрипта в krea-work/test_krea_tool.py — тест [16].
#  3. Выяснено, почему inspect_image_sources показывал chat_files: [] — Open WebUI
#     0.11.3 НАМЕРЕННО не кладёт картинки в payload.files (Chat.svelte фильтрует
#     и chatFiles, и files по условию "!(content_type ?? '').startsWith('image/')"),
#     поэтому у картинок __files__ всегда пуст.
#
# ЧТО ИСПРАВЛЕНО В 2.3 (найдено боем 20.09.2026, проверка E; в v2.2 вывод был неверный):
#  4. В v2.2 решили, что рабочая история — __messages__, и читали там msg['files'].
#     Бой показал chat_files: [] — и это правильно: бэкенд OW 0.11.3 вырезает и их.
#     middleware.py ~2431–2453: картинки переносятся в content частями
#     {'type':'image_url','image_url':{'url': …}}, после чего идёт
#     `message.pop('files', None)`, а инструменту отдаётся уже этот form_data['messages']
#     (строки ~2923 / ~3206 / ~5702). Итог: files нет ни в __files__, ни в __messages__.
#  5. v2.3: вложения ищутся ещё и в content-частях image_url (id файла,
#     /api/v1/files/<id>/content, URL или data URI) — «пустой image=» снова означает
#     «первое изображение сообщения». Закреплено тестом [17].
#
# ВАЖНО ДЛЯ АГЕНТА: разделы «ПОРЯДОК ДЕЙСТВИЙ», «ТАЙМАУТЫ И job_id» и
# «КАК ПРАВИЛЬНО ПОЛУЧАТЬ ИЗОБРАЖЕНИЯ» вписаны в описания методов — читай их.
# =============================================================================

import asyncio
import base64
import difflib
import io
import json
import os
import re
import threading
import time
from collections import OrderedDict

import requests

from pydantic import BaseModel, Field

BASE_URL = "https://api.krea.ai"

# --- сетевые политики (секунды) ----------------------------------------------
HTTP_CONNECT_TIMEOUT_S = 10          # установка соединения
HTTP_READ_TIMEOUT_S = 60             # ожидание ответа на один запрос
SUBMIT_READ_TIMEOUT_S = 90           # POST /generate: приём job_id может быть небыстрым
POLL_INTERVAL_S = 2.0                # первый интервал опроса
POLL_MAX_INTERVAL_S = 8.0            # потолок интервала опроса
MAX_WAIT_S = 300                     # жёсткий потолок ожидания внутри одного вызова
MAX_POLL_ERRORS = 3                  # сколько подряд сетевых сбоев терпим при опросе
RETRY_ATTEMPTS = 3                   # попытки для идемпотентных (GET) запросов
RETRY_BACKOFF_S = 0.8                # база экспоненциального бэкоффа
REGISTRY_LIMIT = 50                  # сколько последних задач помним локально
DEFAULT_MAX_REFS = 10                # если в спеке нет maxItems

TERMINAL_STATES = ("completed", "failed", "cancelled")
ACTIVE_STATES = ("backlogged", "queued", "scheduled", "processing", "sampling")
JOB_ID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                       r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

# --- алиасы моделей ----------------------------------------------------------
# Полный список полей/лимитов/цен каждой модели — в MODEL_SPEC (генерируется).
MODEL_ROUTES = {
    "seedream-4": "image/bytedance/seedream-4",
    "seedream-5-pro": "image/bytedance/seedream-5-pro",
    "seedream-5-lite": "image/bytedance/seedream-5-lite",
    "seededit": "image/bytedance/seededit",
    "qwen-2512": "image/qwen/2512",
    "krea-2-large": "image/krea/krea-2/large",
    "krea-2-medium": "image/krea/krea-2/medium",
    "krea-2-turbo": "image/krea/krea-2/medium-turbo",
    "nano-banana": "image/google/nano-banana",
    "nano-banana-2": "image/google/nano-banana-2",
    "nano-banana-pro": "image/google/nano-banana-pro",
    "nano-banana-flash-lite": "image/google/nano-banana-flash-lite",
    "gpt-image": "image/openai/gpt-image",
    "gpt-image-2": "image/openai/gpt-image-2",
    "gpt-image-2.5-flare": "image/openai/gpt-image-2.5-flare",
    "gpt-image-2.5-sunburst": "image/openai/gpt-image-2.5-sunburst",
    "flux": "image/bfl/flux-1-dev",
    "flux-1.1-pro": "image/bfl/flux-1.1-pro",
    "flux-1.1-ultra": "image/bfl/flux-1.1-pro-ultra",
    "flux-kontext": "image/bfl/flux-1-kontext-dev",
    "ideogram-3": "image/ideogram/ideogram-3",
    "ideogram-2-turbo": "image/ideogram/ideogram-2-turbo",
    "grok-image": "image/xai/grok-imagine-2",
    "grok-edit": "image/xai/grok-imagine-2-edit",
    "z-image": "image/z-image/z-image",
    "muse-image": "image/meta/muse-image",
    "luma-uni-1": "image/luma/uni-1",
    "runway-gen-4": "image/runway/gen-4-image",
    "krea-enhance": "enhance/krea/enhance",
    "topaz-generative": "enhance/topaz/generative-enhance",
}

# >>> GENERATED MODEL_SPEC
# МОДЕЛЬНЫЙ КАТАЛОГ — сгенерирован из OpenAPI-спеки Krea (krea-work/build_model_spec.py).
# Обновление:
#   curl -s https://api.krea.ai/openapi.json -o /tmp/krea_openapi.json
#   python3 krea-work/build_model_spec.py /tmp/krea_openapi.json > krea-work/model_spec_block.py
#   python3 krea-work/splice_spec.py
MODEL_SPEC = {
    "enhance/krea/enhance": {
        "geometry": "none",
        "fields": ["ai_strength", "clarity_strength", "image_scaling_factor", "image_url", "prompt", "rescale_color", "resemblance_strength", "seed", "sharpness"],
        "required": ["image_url"],
        "refs": {"single": ["image_url"]},
    },
    "enhance/krea/legacy-enhance": {
        "geometry": "none",
        "fields": ["ai_strength", "clarity_strength", "image_scaling_factor", "image_url", "prompt", "rescale_color", "resemblance_strength", "scene_image_url", "scene_prompt", "scene_strength", "scene_transfer", "switch_background", "upscaling_activated"],
        "required": ["image_url"],
        "refs": {"single": ["image_url"]},
    },
    "enhance/topaz/bloom-2-enhance": {
        "geometry": "wh",
        "fields": ["autoprompt", "color_preservation", "creativity", "crop_to_fill", "face_preservation", "grain", "grain_density", "grain_model", "grain_size", "grain_strength", "height", "image_scaling_factor", "image_url", "output_format", "prompt", "reference_uri", "seed", "upscaling_activated", "width"],
        "required": ["height", "image_url", "width"],
        "size_bounds": {"w": [1, 10000], "h": [1, 10000]},
        "refs": {"single": ["image_url"]},
    },
    "enhance/topaz/bloom-enhance": {
        "geometry": "wh",
        "fields": ["color_preservation", "creativity", "crop_to_fill", "face_preservation", "height", "image_scaling_factor", "image_url", "output_format", "prompt", "seed", "upscaling_activated", "width"],
        "required": ["height", "image_url", "width"],
        "size_bounds": {"w": [1, 10000], "h": [1, 10000]},
        "refs": {"single": ["image_url"]},
    },
    "enhance/topaz/generative-enhance": {
        "geometry": "wh",
        "fields": ["creativity", "crop_to_fill", "denoise", "detail", "face_enhancement", "face_enhancement_creativity", "face_enhancement_strength", "height", "image_scaling_factor", "image_url", "model", "output_format", "prompt", "seed", "sharpen", "subject_detection", "texture", "upscaling_activated", "width"],
        "required": ["height", "image_url", "width"],
        "size_bounds": {"w": [1, 32000], "h": [1, 32000]},
        "refs": {"single": ["image_url"]},
    },
    "enhance/topaz/standard-enhance": {
        "geometry": "wh",
        "fields": ["crop_to_fill", "denoise", "face_enhancement", "face_enhancement_creativity", "face_enhancement_strength", "fix_compression", "height", "image_scaling_factor", "image_url", "model", "output_format", "prompt", "seed", "sharpen", "strength", "subject_detection", "upscaling_activated", "width"],
        "required": ["height", "image_url", "model", "width"],
        "size_bounds": {"w": [1, 32000], "h": [1, 32000]},
        "refs": {"single": ["image_url"]},
    },
    "enhance/topaz/wonder-35-enhance": {
        "geometry": "wh",
        "fields": ["crop_to_fill", "enhancement_strength", "grain", "grain_density", "grain_model", "grain_size", "grain_strength", "height", "image_scaling_factor", "image_url", "output_format", "seed", "upscaling_activated", "width"],
        "required": ["height", "image_url", "width"],
        "size_bounds": {"w": [1, 16000], "h": [1, 16000]},
        "refs": {"single": ["image_url"]},
    },
    "image/bfl/flux-1-dev": {
        "geometry": "wh",
        "fields": ["guidance_scale", "height", "image_style_references", "image_url", "prompt", "seed", "steps", "strength", "style_images", "styles", "width"],
        "required": ["prompt"],
        "size_bounds": {"w": [512, 2368], "h": [512, 2368]},
        "refs": {"single": ["image_url"], "style_images": 10, "image_style_references": 10},
        "price_min": 0.007,
        "price_max": 0.007,
    },
    "image/bfl/flux-1-kontext-dev": {
        "geometry": "wh",
        "fields": ["guidance_scale", "height", "image_url", "prompt", "seed", "steps", "strength", "style_images", "width"],
        "required": ["prompt"],
        "size_bounds": {"w": [512, 8192], "h": [512, 8192]},
        "refs": {"single": ["image_url"], "style_images": 1},
        "price_min": 0.021,
        "price_max": 0.021,
    },
    "image/bfl/flux-1.1-pro": {
        "geometry": "wh",
        "fields": ["height", "prompt", "seed", "width"],
        "required": ["height", "prompt", "width"],
        "size_bounds": {"w": [256, 1440], "h": [256, 1440]},
        "price_min": 0.042,
        "price_max": 0.042,
    },
    "image/bfl/flux-1.1-pro-ultra": {
        "geometry": "wh",
        "fields": ["height", "prompt", "raw", "seed", "width"],
        "required": ["prompt"],
        "size_bounds": {"w": [512, 8192], "h": [512, 8192]},
        "price_min": 0.063,
        "price_max": 0.063,
    },
    "image/bytedance/seededit": {
        "geometry": "none",
        "fields": ["image_url", "prompt", "seed"],
        "required": ["image_url", "prompt"],
        "refs": {"single": ["image_url"]},
    },
    "image/bytedance/seedream-4": {
        "geometry": "wh",
        "fields": ["height", "prompt", "seed", "style_images", "width"],
        "required": ["height", "prompt", "width"],
        "size_bounds": {"w": [None, 4096], "h": [None, 4096]},
        "refs": {"style_images": 10},
        "price_min": 0.0315,
        "price_max": 0.0315,
    },
    "image/bytedance/seedream-5-lite": {
        "geometry": "wh",
        "fields": ["height", "prompt", "seed", "style_images", "width"],
        "required": ["height", "prompt", "width"],
        "size_bounds": {"w": [None, 4096], "h": [None, 4096]},
        "refs": {"style_images": 14},
        "price_min": 0.0368,
        "price_max": 0.0368,
    },
    "image/bytedance/seedream-5-pro": {
        "geometry": "wh",
        "fields": ["height", "prompt", "seed", "style_images", "width"],
        "required": ["height", "prompt", "width"],
        "size_bounds": {"w": [None, 4096], "h": [None, 4096]},
        "refs": {"style_images": 10},
        "price_min": 0.0473,
        "price_max": 0.1229,
        "price_tiers": "{\"resolutionTier\": \"1.5k\", \"referenceImageCount\": 0}=0.0473; {\"resolutionTier\": \"1.5k\", \"referenceImageCount\": 1}=0.0473; {\"resolutionTier\": \"1.5k\", \"referenceImageCount\": 2}=0.0504; {\"resolutionTier\": \"1.5k\", \"reference",
    },
    "image/google/nano-banana": {
        "geometry": "both",
        "fields": ["aspect_ratio", "height", "image_urls", "prompt", "width"],
        "required": ["prompt"],
        "aspect_ratios": ["21:9", "1:1", "4:3", "3:2", "2:3", "5:4", "4:5", "3:4", "16:9", "9:16"],
        "refs": {"image_urls": 10},
        "price_min": 0.0429,
        "price_max": 0.0429,
    },
    "image/google/nano-banana-2": {
        "geometry": "both",
        "fields": ["aspect_ratio", "height", "image_urls", "prompt", "resolution", "width"],
        "required": ["prompt"],
        "aspect_ratios": ["4:1", "21:9", "1:1", "4:3", "3:2", "2:3", "5:4", "4:5", "3:4", "16:9", "9:16", "1:4", "1:8"],
        "resolutions": ["1K", "2K", "4K"],
        "resolution_default": "1K",
        "refs": {"image_urls": 10},
        "price_min": 0.08,
        "price_max": 0.16,
        "price_tiers": "{\"resolution\": \"1K\"}=0.08; {\"resolution\": \"2K\"}=0.12; {\"resolution\": \"4K\"}=0.16",
    },
    "image/google/nano-banana-flash-lite": {
        "geometry": "both",
        "fields": ["aspect_ratio", "height", "image_urls", "prompt", "width"],
        "required": ["prompt"],
        "aspect_ratios": ["21:9", "1:1", "4:3", "3:2", "2:3", "5:4", "4:5", "3:4", "16:9", "9:16"],
        "refs": {"image_urls": 14},
        "price_min": 0.034,
        "price_max": 0.034,
    },
    "image/google/nano-banana-pro": {
        "geometry": "both",
        "fields": ["aspect_ratio", "height", "image_urls", "prompt", "resolution", "width"],
        "required": ["prompt"],
        "aspect_ratios": ["21:9", "1:1", "4:3", "3:2", "2:3", "5:4", "4:5", "3:4", "16:9", "9:16"],
        "resolutions": ["1K", "2K", "4K"],
        "resolution_default": "1K",
        "refs": {"image_urls": 10},
        "price_min": 0.15,
        "price_max": 0.3,
        "price_tiers": "{\"resolution\": \"1K\"}=0.15; {\"resolution\": \"2K\"}=0.15; {\"resolution\": \"4K\"}=0.3",
    },
    "image/ideogram/ideogram-2-turbo": {
        "geometry": "wh",
        "fields": ["height", "prompt", "seed", "width"],
        "required": ["prompt"],
        "size_bounds": {"w": [512, 8192], "h": [512, 8192]},
        "price_min": 0.0263,
        "price_max": 0.0263,
    },
    "image/ideogram/ideogram-3": {
        "geometry": "wh",
        "fields": ["character_reference_images", "height", "prompt", "seed", "style_images", "width"],
        "required": ["prompt"],
        "size_bounds": {"w": [512, 8192], "h": [512, 8192]},
        "refs": {"style_images": 10, "character_reference_images": 10},
        "price_min": 0.063,
        "price_max": 0.1575,
        "price_tiers": "{\"hasCharacterRefs\": false}=0.063; {\"hasCharacterRefs\": true}=0.1575",
    },
    "image/krea/krea-2/large": {
        "geometry": "ar",
        "fields": ["aspect_ratio", "complexity", "creativity", "image_style_references", "image_url", "intensity", "moodboards", "movement", "prompt", "resolution", "seed", "strength", "styles"],
        "required": ["aspect_ratio", "prompt", "resolution"],
        "aspect_ratios": ["1:1", "4:3", "3:2", "16:9", "2.35:1", "4:5", "3:4", "2:3", "9:16"],
        "resolutions": ["1K"],
        "resolution_default": "1K",
        "refs": {"single": ["image_url"], "image_style_references": 10},
        "price_min": 0.06,
        "price_max": 0.07,
        "price_tiers": "{\"k2BillingTier\": \"text-to-image\"}=0.06; {\"k2BillingTier\": \"srefs\"}=0.065; {\"k2BillingTier\": \"moodboards\"}=0.07",
    },
    "image/krea/krea-2/medium": {
        "geometry": "ar",
        "fields": ["aspect_ratio", "complexity", "creativity", "image_style_references", "image_url", "intensity", "moodboards", "movement", "prompt", "resolution", "seed", "strength", "styles"],
        "required": ["aspect_ratio", "prompt", "resolution"],
        "aspect_ratios": ["1:1", "4:3", "3:2", "16:9", "2.35:1", "4:5", "3:4", "2:3", "9:16"],
        "resolutions": ["1K"],
        "resolution_default": "1K",
        "refs": {"single": ["image_url"], "image_style_references": 10},
        "price_min": 0.03,
        "price_max": 0.04,
        "price_tiers": "{\"k2BillingTier\": \"text-to-image\"}=0.03; {\"k2BillingTier\": \"srefs\"}=0.035; {\"k2BillingTier\": \"moodboards\"}=0.04",
    },
    "image/krea/krea-2/medium-turbo": {
        "geometry": "ar",
        "fields": ["aspect_ratio", "complexity", "creativity", "image_style_references", "image_url", "intensity", "moodboards", "movement", "prompt", "resolution", "seed", "strength", "styles"],
        "required": ["aspect_ratio", "prompt", "resolution"],
        "aspect_ratios": ["1:1", "4:3", "3:2", "16:9", "2.35:1", "4:5", "3:4", "2:3", "9:16"],
        "resolutions": ["1K"],
        "resolution_default": "1K",
        "refs": {"single": ["image_url"], "image_style_references": 10},
        "price_min": 0.015,
        "price_max": 0.02,
        "price_tiers": "{\"k2BillingTier\": \"text-to-image\"}=0.015; {\"k2BillingTier\": \"srefs\"}=0.0175; {\"k2BillingTier\": \"moodboards\"}=0.02",
    },
    "image/luma/uni-1": {
        "geometry": "wh",
        "fields": ["height", "mode", "output_format", "prompt", "style", "style_images", "web_search", "width"],
        "required": ["prompt"],
        "size_bounds": {"w": [512, 8192], "h": [512, 8192]},
        "refs": {"style_images": 9},
        "price_min": 0.0404,
        "price_max": 0.127,
        "price_tiers": "{\"mode\": \"standard\", \"referenceImageCount\": 0}=0.0404; {\"mode\": \"standard\", \"referenceImageCount\": 1}=0.0434; {\"mode\": \"standard\", \"referenceImageCount\": 2}=0.0464; {\"mode\": \"standard\", \"referenceImageCount\": 3}=0.0494; ",
    },
    "image/meta/muse-image": {
        "geometry": "ar_opt",
        "fields": ["aspect_ratio", "prompt", "style_images"],
        "required": ["prompt"],
        "aspect_ratios": ["21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"],
        "refs": {"style_images": 10},
        "price_min": 0.01,
        "price_max": 0.01,
    },
    "image/openai/gpt-image": {
        "geometry": "wh",
        "fields": ["height", "image_urls", "prompt", "quality", "styles", "width"],
        "required": ["prompt"],
        "size_bounds": {"w": [512, 8192], "h": [512, 8192]},
        "quality": ["low", "medium", "high", "auto"],
        "refs": {"image_urls": 15},
        "price_min": 0.3747,
        "price_max": 0.3747,
    },
    "image/openai/gpt-image-2": {
        "geometry": "both",
        "fields": ["aspect_ratio", "height", "image_urls", "prompt", "quality", "resolution", "width"],
        "required": ["prompt"],
        "aspect_ratios": ["16:9", "2:1", "3:2", "4:3", "1:1", "3:4", "2:3", "1:2", "9:16"],
        "resolutions": ["1K", "2K", "4K"],
        "quality": ["low", "medium", "high", "auto"],
        "refs": {"image_urls": 10},
    },
    "image/openai/gpt-image-2.5-flare": {
        "geometry": "ar",
        "fields": ["aspect_ratio", "background", "image_urls", "prompt", "quality", "resolution"],
        "required": ["aspect_ratio", "prompt", "resolution"],
        "aspect_ratios": ["16:9", "2:1", "3:2", "4:3", "1:1", "3:4", "2:3", "1:2", "9:16"],
        "resolutions": ["1K", "2K", "4K"],
        "resolution_default": "1K",
        "quality": ["low", "medium", "high", "xhigh", "max"],
        "refs": {"image_urls": 10},
    },
    "image/openai/gpt-image-2.5-sunburst": {
        "geometry": "ar",
        "fields": ["aspect_ratio", "image_urls", "prompt", "quality", "resolution"],
        "required": ["aspect_ratio", "prompt", "resolution"],
        "aspect_ratios": ["16:9", "2:1", "3:2", "4:3", "1:1", "3:4", "2:3", "1:2", "9:16"],
        "resolutions": ["1K", "2K", "4K"],
        "resolution_default": "1K",
        "quality": ["low", "medium", "high", "xhigh", "max"],
        "refs": {"image_urls": 10},
    },
    "image/qwen/2512": {
        "geometry": "wh",
        "fields": ["cfg_scale", "height", "negative_prompt", "num_inference_steps", "prompt", "seed", "styles", "width"],
        "required": ["prompt"],
        "size_bounds": {"w": [256, 4096], "h": [256, 4096]},
    },
    "image/runway/gen-4-image": {
        "geometry": "wh",
        "fields": ["height", "prompt", "reference_images", "seed", "width"],
        "required": ["prompt", "reference_images"],
        "size_bounds": {"w": [512, 8192], "h": [512, 8192]},
        "refs": {"reference_images": 3},
    },
    "image/xai/grok-imagine-2": {
        "geometry": "ar",
        "fields": ["aspect_ratio", "prompt", "quality", "resolution", "style_images"],
        "required": ["aspect_ratio", "prompt", "resolution"],
        "aspect_ratios": ["2:1", "20:9", "19.5:9", "16:9", "3:2", "4:3", "1:1", "3:4", "2:3", "9:16", "9:19.5", "9:20", "1:2"],
        "resolutions": ["1K", "2K"],
        "resolution_default": "1K",
        "quality": ["low", "medium"],
        "refs": {"style_images": 3},
        "price_min": 0.042,
        "price_max": 0.1155,
        "price_tiers": "{\"quality\": \"low\", \"resolution\": \"1K\", \"referenceImageCount\": 0}=0.042; {\"quality\": \"low\", \"resolution\": \"1K\", \"referenceImageCount\": 1}=0.0525; {\"quality\": \"low\", \"resolution\": \"1K\", \"referenceImageCount\": 2}=0.063; {\"q",
    },
    "image/xai/grok-imagine-2-edit": {
        "geometry": "none",
        "fields": ["image_urls", "prompt", "quality", "resolution"],
        "required": ["image_urls", "prompt"],
        "resolutions": ["1K", "2K"],
        "resolution_default": "1K",
        "quality": ["low", "medium"],
        "refs": {"image_urls": 3},
        "price_min": 0.042,
        "price_max": 0.1155,
        "price_tiers": "{\"quality\": \"low\", \"resolution\": \"1K\", \"referenceImageCount\": 0}=0.042; {\"quality\": \"low\", \"resolution\": \"1K\", \"referenceImageCount\": 1}=0.0525; {\"quality\": \"low\", \"resolution\": \"1K\", \"referenceImageCount\": 2}=0.063; {\"q",
    },
    "image/z-image/z-image": {
        "geometry": "ar",
        "fields": ["aspect_ratio", "denoising_strength", "image_url", "prompt", "resolution", "seed", "skip_prompt_expansion", "style_images", "styles"],
        "required": ["aspect_ratio", "prompt", "resolution"],
        "aspect_ratios": ["1:1", "4:3", "2:3", "16:9", "9:16"],
        "resolutions": ["1K"],
        "resolution_default": "1K",
        "refs": {"single": ["image_url"], "style_images": 1},
    },
}
# <<< GENERATED MODEL_SPEC

# Геометрия: сколько референсов реально принимает модель (по умолчанию),
# порядок предпочтения поля для референсов при генерации.
REF_FIELD_ORDER = ("image_urls", "image_style_references", "style_images",
                   "character_reference_images", "reference_images")
# поля-референсы, которые принимают не URL, а id стиля аккаунта — картинку туда не кладём
NON_URL_REF_FIELDS = ("styles", "moodboards")

IMAGE_MODEL_SELECTOR = (
    "  Выбор модели (параметр `model`):\n"
    "  - если пользователь не назвал модель — НЕ передавай этот параметр, возьмётся дефолт (seedream-4);\n"
    "  - 'seedream-4' — фотореализм + надёжный текст на картинке, $0.0315 (рабочая лошадка);\n"
    "  - 'flux-1-dev' ($0.007) и 'muse-image' ($0.01) — самые дешёвые, для черновиков и проб;\n"
    "  - 'krea-2-turbo' ($0.015) / 'krea-2-medium' ($0.03) / 'krea-2-large' ($0.06) — самые\n"
    "    эстетичные выразительные картинки (арт, обложки), но только aspect_ratio+resolution;\n"
    "  - 'nano-banana-pro' ($0.15–0.3) / 'gpt-image-2' — «умные» модели: сложные промпты,\n"
    "    референсные фото людей/объектов, надписи, композиции (передавай им image_refs);\n"
    "  - 'gpt-image' — $0.3747, заметно дороже: не выбирай его без причины;\n"
    "  - 'flux-1.1-ultra' ($0.063), 'ideogram-3' ($0.063), 'grok-image' ($0.042) — стиль,\n"
    "    бренды, скорость. Точные поля, лимиты и цены любой модели — метод list_models.\n"
    "  Допустим и полный Krea id: 'google/nano-banana-pro', 'image/bfl/flux-1-kontext-dev'."
)

# ---------------------------------------------------------------------------
# Единая инструкция «КАК ПРАВИЛЬНО ПОЛУЧАТЬ ФАЙЛЫ». Вставляется в docstring
# каждого метода, работающего с изображениями (маркер [FILE_SOURCES]).
# ---------------------------------------------------------------------------
FILE_SOURCES = """КАК ПРАВИЛЬНО ПОЛУЧАТЬ ИЗОБРАЖЕНИЯ (обязательный раздел):
  Krea принимает картинки ТОЛЬКО как: https-URL, asset URL от POST /assets или
  data URI. «Путь к файлу» для Krea ничего не значит — байты сначала нужно добыть.
  Этот инструмент делает добычу сам, тебе нужно передать лишь ссылку-указатель:
  1) КАРТИНКА В СООБЩЕНИИ ПОЛЬЗОВАТЕЛЯ (вложение чата). Ты видишь её как изображение
     и/или упоминание с id файла (attached_files / list_chat_files).
     ДЕЙСТВИЕ: передай этот id КАК ЕСТЬ в параметр image / image_refs
     (пример: image="9f2c...-uuid") — инструмент скачает байты через внутренний API
     Open WebUI (GET /api/v1/files/{id}/content) под сессией пользователя.
     ВАЖНО: встроенный view_file для картинок возвращает ПУСТОЙ текст — им байты не
     получить, поэтому не «читай» картинку, а передай id сюда.
  2) КАРТИНКА ФАЙЛОМ В ФАЙЛОВОЙ СИСТЕМЕ (Open Terminal / «Файловое хранилище»).
     ДЕЙСТВИЕ: передай АБСОЛЮТНЫЙ путь (пример: /home/user/photo.png) в image /
     image_refs — инструмент прочитает байты через HTTP API хранилища.
     Путь бери реальный (из list_files/read_file этого хранилища), не выдумывай.
  3) КАРТИНКА УЖЕ В ИНТЕРНЕТЕ ИЛИ ЗАГРУЖЕНА: передай https-URL или asset URL от
     upload_asset КАК ЕСТЬ — скачивать ничего не нужно.
  4) НЕСКОЛЬКО КАРТИНОК (коллаж, референсы): список указателей в image_refs — каждый
     элемент разрешается по своим правилам (id, путь, URL).
  КАК ЭТО РАБОТАЕТ ВНУТРИ (важно для больших файлов): инструмент скачивает байты с
  жёстким лимитом (Valves → max_media_bytes), а затем, если картинка больше
  inline_data_uri_max_bytes, ЗАЛИВАЕТ её на Krea (POST /assets) и подставляет
  короткий URL. Причина — экономия памяти и размера тела: data URI в 1.33 раза длиннее
  самих байт и существует в 3–4 копиях. Лимит maxLength=1024 у media-полей Krea
  НЕ проверяет (проверено 20.09.2026, krea-plan.md, проверка A).
  ЗАПРЕЩЕНО: передавать путь в надежде, что Krea сам его откроет; передавать вывод
  view_file/read_file как «содержимое картинки»; вручную склеивать огромные data URI
  в параметрах — доверь это инструменту.
  ПРО «ПУСТОЙ image=»: инструмент сам ищет первое изображение сообщения. Оба очевидных
  источника Open WebUI 0.11.3 урезает сам (проверено 20.09.2026): в payload.files
  картинок НЕТ (фронтенд фильтрует их по content_type), а в __messages__ НЕТ поля
  files (бэкенд перед вызовом инструмента делает message.pop('files', None)), зато
  картинка лежит в content частью {'type':'image_url','image_url':{'url': …}}.
  Инструмент разбирает все три вида: __files__, msg['files'] и content-части image_url.
  Если ничего не нашлось — передавай указатель явно (id, путь или URL)."""

# ---------------------------------------------------------------------------
# Инструкция про таймауты/деньги/job_id (маркер [AGENT_NOTES]).
# ---------------------------------------------------------------------------
AGENT_NOTES = """ТАЙМАУТЫ, ДЕНЬГИ И job_id (обязательный раздел, читай перед вызовом):
  ПОРЯДОК ДЕЙСТВИЙ:
    1) сомневаешься, жив ли сервис/ключ — вызови krea_status (бесплатно, без кредитов);
    2) не уверен в полях/цене модели — list_models(model="...") (локально, без сети);
    3) генерация/правка/апскейл — generate_image / edit_image / enhance_image;
    4) если вернулось «ещё выполняется» — get_job(job_id="..."). НЕ запускай генерацию
       повторно: каждый POST /generate — это ещё один платный запрос;
    5) если job_id потерялся (модель перепутала символы, ответ обрезан) — find_jobs:
       он найдёт задачу по времени создания и вернёт её id и ссылки.
  ПРО ТАЙМАУТЫ:
    - wait_s — сколько ЖДАТЬ внутри вызова (по умолчанию 180, потолок 300). Это не
      «лимит Krea»: задача продолжает считаться и после возврата управления;
    - вызов может вернуть управление раньше срока («ещё выполняется») — это НОРМАЛЬНО;
    - не проси wait_s больше 300: инструмент всё равно обрежет, а долгое ожидание
      упирается в таймауты HTTP-фронта Open WebUI;
    - типичные времена: текст→картинка 10–60 c, апскейл/топаз 1–5 мин, 4K дольше.
  ПРО ОШИБКИ KREA:
    - 401 — неверный/просроченный api_key (проверь Valves → api_key);
    - 402 — на аккаунте кончились средства;
    - 400 — какое-то поле не подходит модели (сверься с list_models; лишнее поле
      запрещено: в схемах Krea additionalProperties=false);
    - 429 «maximum number of concurrent jobs» — слишком много одновременных задач:
      подожди, посмотри активные в krea_status, не создавай новые пачками;
    - 5xx — сбой на стороне Krea, задача могла не создаться: проверь find_jobs.
  ПРО ДЕНЬГИ: перед пачкой вариантов выбери дешёвую модель для черновиков, а дорогую
  (gpt-image $0.37) — только под итоговый кадр. Повторный вызов того же промпта без
  надобности = оплата ещё раз."""


class MediaError(Exception):
    """Не удалось превратить указатель пользователя в байты/URL для Krea."""


class KreaApiError(Exception):
    """Ошибка/недоступность Krea API с уже готовой подсказкой для человека."""

    def __init__(self, message: str, status: int = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class KreaTransport(Exception):
    """Сетевой сбой (таймаут/обрыв) — задача после такого могла уже создаться."""


# ---------------------------------------------------------------------------
# Мелкие чистые помощники
# ---------------------------------------------------------------------------

def _route(model: str) -> str:
    """Alias или полный id → route вида 'image/provider/name'."""
    key = (model or "").strip().strip("/")
    if not key:
        raise ValueError("Не указана модель.")
    if key in MODEL_ROUTES:
        return MODEL_ROUTES[key]
    if key in MODEL_SPEC:
        return key
    if "/" in key:
        if key.startswith(("video/", "3d/", "audio/")):
            raise ValueError(
                f"'{key}' — это не картинка, а {key.split('/')[0]}. "
                "Этот инструмент работает только с image/ и enhance/.")
        if key.startswith(("image/", "enhance/")):
            raise ValueError(
                f"Модель '{key}' не поддерживается (нет в списке доступных image/enhance-моделей). "
                f"Возможно, ты имел в виду: {_suggest(key)}")
        for cat in ("image/", "enhance/"):
            if cat + key in MODEL_SPEC:
                return cat + key
        raise ValueError(f"Неизвестная модель '{key}'. Похожие: {_suggest(key)}")
    raise ValueError(f"Неизвестная модель '{key}'. Похожие: {_suggest(key)}. "
                     "Полный список алиасов, цен и лимитов — метод list_models.")


def _suggest(key: str) -> str:
    pool = list(MODEL_ROUTES) + list(MODEL_SPEC)
    close = difflib.get_close_matches(key, pool, n=3, cutoff=0.5)
    return ", ".join(close) if close else "—"


def _known(route: str) -> str:
    for alias, r in MODEL_ROUTES.items():
        if r == route:
            return alias
    return route


def _spec(route: str) -> dict:
    return MODEL_SPEC.get(route) or {}


def _fields(route: str) -> set:
    return set(_spec(route).get("fields") or ())


def _is_image_bytes(data: bytes) -> bool:
    return bool(data) and (
        data[:8].startswith(b"\x89PNG\r\n\x1a\n") or data[:3] == b"\xff\xd8\xff"
        or (data[:4] == b"RIFF" and data[8:12] == b"WEBP") or data[:4] == b"GIF8"
        or data[:2] == b"BM" or data[:4] == b"\x00\x00\x01\x00")


def _mime_by_magic(data: bytes) -> str:
    if data[:8].startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:4] == b"GIF8":
        return "image/gif"
    return "image/png"


def _mime_by_name(name: str) -> str:
    low = (name or "").lower()
    for ext, mime in ((".jpg", "image/jpeg"), (".jpeg", "image/jpeg"), (".webp", "image/webp"),
                      (".gif", "image/gif"), (".png", "image/png")):
        if low.endswith(ext):
            return mime
    return "image/png"


def _data_uri(data: bytes, hint_name: str = "") -> str:
    mime = _mime_by_magic(data)
    if mime == "image/png" and hint_name:
        mime = _mime_by_name(hint_name) or mime
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


# Картинка внутри текста сообщения: data URI или ссылка OW на файл (markdown-вставка)
_CONTENT_MEDIA_RE = re.compile(
    r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]{64,}"
    r"|/api/v1/files/[0-9a-fA-F\-]{20,64}/content",
    re.I,
)


def _looks_like_file_id(value: str) -> bool:
    """Open WebUI file id — uuid-подобная строка без '/' и '.'."""
    v = (value or "").strip()
    if v.lower().startswith("fileid:"):
        return True
    if "/" in v or "\\" in v or "." in v:
        return False
    return bool(re.fullmatch(r"[0-9a-fA-F\-]{20,64}", v))


def _b64_prefix_bytes(uri: str, max_bytes: int = 96 * 1024) -> bytes:
    """Первые max_bytes байт из data URI — без декодирования всего файла."""
    try:
        payload = uri.split(",", 1)[1]
    except IndexError:
        return b""
    limit = min(len(payload), ((max_bytes + 2) // 3) * 4)
    chunk = payload[:limit]
    chunk = chunk[: len(chunk) - (len(chunk) % 4)]
    try:
        return base64.b64decode(chunk)
    except Exception:
        return b""


def _size_from_header(data: bytes):
    """(width, height) по первым байтам картинки; None, если не распознали."""
    import struct
    if not data:
        return None
    try:
        if data[:8].startswith(b"\x89PNG\r\n\x1a\n"):
            return struct.unpack(">II", data[16:24])
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            if data[12:16] == b"VP8X":
                return 1 + int.from_bytes(data[24:27], "little"), 1 + int.from_bytes(data[27:30], "little")
            if data[12:16] == b"VP8L":
                b = data[21:25]
                return (1 + (((b[1] & 0x3F) << 8) | b[0]),
                        1 + (((b[3] & 0x0F) << 10) | (b[2] << 2) | ((b[1] & 0xC0) >> 6)))
        if data[:3] == b"\xff\xd8\xff":
            i = 2
            while i < len(data) - 9:
                if data[i] != 0xFF:
                    i += 1
                    continue
                m = data[i + 1]
                if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    h, w = struct.unpack(">HH", data[i + 5:i + 9])
                    return w, h
                i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    except Exception:
        return None
    return None


def _media_size(media: str):
    """(width, height) из data URI по первым килобайтам — без полной распаковки."""
    if not isinstance(media, str) or not media.startswith("data:"):
        return None
    return _size_from_header(_b64_prefix_bytes(media))


def _asset_geometry(payload: dict):
    """(width, height) из ответа POST /assets — источник истины с 20.09.2026 (проверка B).

    Возвращает None, если полей нет или они не числа/нули: тогда вызывающий код берёт
    размер из локального заголовка файла (`_size_from_header`).
    """
    if not isinstance(payload, dict):
        return None
    try:
        w, h = int(payload.get("width") or 0), int(payload.get("height") or 0)
    except (TypeError, ValueError):
        return None
    return (w, h) if w > 0 and h > 0 else None


def _asset_size_bytes(payload: dict):
    """size_bytes из ответа POST /assets (или None)."""
    try:
        value = int((payload or {}).get("size_bytes"))
    except (TypeError, ValueError, AttributeError):
        return None
    return value if value > 0 else None


def _looks_like_asset_url(url: str) -> bool:
    return isinstance(url, str) and url.startswith("http") and "krea.ai" in url


def _extract_urls(job: dict) -> list:
    """result.urls бывает списком строк, списком {type,url} или мапой name→url."""
    res = (job or {}).get("result") or {}
    urls = res.get("urls") if isinstance(res, dict) else None
    out = []
    if isinstance(urls, list):
        for u in urls:
            if isinstance(u, str):
                out.append(u)
            elif isinstance(u, dict) and u.get("url"):
                out.append(u["url"])
    elif isinstance(urls, dict):
        for v in urls.values():
            if isinstance(v, str):
                out.append(v)
    if not out and isinstance(res, dict):
        for v in res.values():
            if isinstance(v, list) and v and isinstance(v[0], str) and v[0].startswith("http"):
                out.extend(v)
                break
    return out


def _job_error(job: dict) -> str:
    err = (job or {}).get("error") or {}
    if isinstance(err, dict):
        msg = err.get("message") or err.get("code") or ""
        code = err.get("code") or ""
        return (f"{code}: {msg}" if msg and code and msg != code else (msg or code or "")).strip()
    return str(err or "")


def _http_hint(status: int, body: str = "") -> str:
    hints = {
        400: "Krea отклонил запрос (400): какое-то поле не подходит модели. Сверь набор полей "
             "с list_models(model='...') — лишние поля запрещены (additionalProperties=false).",
        401: "Неверный или отозванный API-ключ Krea (401). Проверь Valves → api_key.",
        402: "На аккаунте Krea закончились средства (402). Пополни баланс или возьми дешёвую модель.",
        403: "Krea вернул 403: тариф/права не позволяют операцию.",
        404: "Krea вернул 404: не найден job_id/asset или модели нет. Проверь id через find_jobs.",
        413: "Krea вернул 413: файл слишком большой. Уменьши размер картинки.",
        422: "Krea вернул 422: схема запроса не прошла валидацию.",
        429: "Krea вернул 429 «maximum number of concurrent jobs»: слишком много одновременных "
             "задач. Посмотри активные в krea_status, дождись завершения и повтори — НЕ создавай "
             "новые пачки задач.",
        500: "Внутренняя ошибка Krea (500). Повтори позже; задача могла не создаться — проверь find_jobs.",
        502: "Krea временно недоступен (502 Bad Gateway). Повтори позже.",
        503: "Krea временно недоступен (503). Повтори позже.",
        504: "Krea не успел ответить (504). Повтори позже; проверь find_jobs, не создалась ли задача.",
    }
    return hints.get(status, f"Krea вернул HTTP {status}.")


def _fmt_dt(iso: str) -> str:
    return (iso or "").replace("T", " ").replace("Z", " UTC")[:19]


def _parse_dt(iso: str):
    from datetime import datetime, timezone
    try:
        return datetime.fromisoformat((iso or "").replace("Z", "+00:00"))
    except Exception:
        return None


def _px_for_ratio(aspect_ratio: str, base: int = 1024):
    """Пиксели для пропорции: короткая сторона = base."""
    try:
        a, b = (float(x) for x in (aspect_ratio or "1:1").split(":"))
        if a <= 0 or b <= 0:
            raise ValueError
    except Exception:
        a, b = 1.0, 1.0
    if a >= b:
        w, h = base * a / b, float(base)
    else:
        w, h = float(base), base * b / a
    return int(round(w / 8) * 8), int(round(h / 8) * 8)


def _closest_aspect(w: float, h: float, allowed: list) -> str:
    """Ближайшая разрешённая пропорция (по лог-разнице сторон)."""
    import math
    if not allowed:
        return "1:1"
    target = math.log(max(w, 1) / max(h, 1))
    best, best_d = allowed[0], 1e9
    for a in allowed:
        try:
            aw, ah = (float(x) for x in a.split(":"))
            d = abs(math.log(aw / ah) - target)
        except Exception:
            continue
        if d < best_d:
            best, best_d = a, d
    return best


def _fit_size(w: int, h: int, bounds: dict, notes: list, label: str = ""):
    """Подогнать px под границы модели (равномерное масштабирование)."""
    if not bounds:
        return int(w), int(h)
    wb = bounds.get("w") or [None, None]
    hb = bounds.get("h") or [None, None]
    lo_w, hi_w, lo_h, hi_h = wb[0], wb[1], hb[0], hb[1]
    f = 1.0
    caps = [c for c in ((hi_w / w if hi_w else None), (hi_h / h if hi_h else None)) if c]
    if caps:
        f = min(f, min(caps))
    floors = [c for c in ((lo_w / w if lo_w else None), (lo_h / h if lo_h else None)) if c]
    if floors:
        f = max(f, max(floors))
    nw, nh = max(1, int(round(w * f))), max(1, int(round(h * f)))
    if hi_w and nw > hi_w:
        nw = int(hi_w)
    if hi_h and nh > hi_h:
        nh = int(hi_h)
    if (nw, nh) != (int(w), int(h)):
        notes.append(f"{label}размер подогнан под лимиты модели "
                     f"{int(w)}x{int(h)} → {nw}x{nh} (границы: "
                     f"{lo_w or '—'}..{hi_w or '—'} x {lo_h or '—'}..{hi_h or '—'}).")
    return nw, nh


def _pick_aspect_ratio(requested: str, allowed: list, notes: list, src=None) -> str:
    allowed = allowed or ["1:1"]
    req = (requested or "1:1").strip()
    if req in allowed:
        return req
    if src:
        fallback = _closest_aspect(src[0], src[1], allowed)
    else:
        try:
            rw, rh = (float(x) for x in (req.split(":") + ["1", "1"])[:2])
        except Exception:
            rw, rh = 1.0, 1.0
        fallback = _closest_aspect(rw or 1.0, rh or 1.0, allowed)
    notes.append(f"модель не поддерживает aspect_ratio={req}; взят ближайший {fallback} "
                 f"(доступно: {', '.join(allowed)}).")
    return fallback


def _pick_resolution(spec: dict, requested: str, notes: list):
    allowed = spec.get("resolutions") or []
    if not allowed:
        return None
    req = (requested or spec.get("resolution_default") or allowed[0]).upper()
    if req in allowed:
        return req
    dflt = spec.get("resolution_default") or allowed[0]
    if req != dflt:
        notes.append(f"модель не поддерживает resolution={req}; взят {dflt} "
                     f"(доступно: {', '.join(allowed)}).")
    return dflt


def _apply_geometry(spec: dict, payload: dict, aspect_ratio: str, w: int, h: int,
                    resolution: str, notes: list, src_size=None) -> None:
    """Заполнить геометрию запроса строго по возможностям модели (иначе 400)."""
    geom = spec.get("geometry", "none")
    fields = set(spec.get("fields") or ())
    ar_allowed = spec.get("aspect_ratios") or []
    bounds = spec.get("size_bounds") or {}
    pxw = pxh = 0
    if w and h:
        pxw, pxh = _fit_size(w, h, bounds, notes)
    elif isinstance(src_size, (tuple, list)) and len(src_size) == 2:
        pxw, pxh = _fit_size(src_size[0], src_size[1], bounds, notes)

    if geom == "ar":
        payload["aspect_ratio"] = _pick_aspect_ratio(aspect_ratio, ar_allowed, notes, src_size)
        res = _pick_resolution(spec, resolution, notes)
        if res:
            payload["resolution"] = res
        if w and h:
            notes.append(f"модель задаёт размер только пропорцией (aspect_ratio="
                         f"{payload['aspect_ratio']}), точные {w}x{h} px не передавались.")
    elif geom == "wh":
        if not (pxw and pxh):
            pxw, pxh = _fit_size(*_px_for_ratio(aspect_ratio), bounds, notes)
        payload["width"], payload["height"] = int(pxw), int(pxh)
        res = _pick_resolution(spec, resolution, notes)
        if res and "resolution" in fields:
            payload["resolution"] = res
    elif geom == "both":
        if pxw and pxh:
            payload["width"], payload["height"] = int(pxw), int(pxh)
        elif ar_allowed:
            payload["aspect_ratio"] = _pick_aspect_ratio(aspect_ratio, ar_allowed, notes, src_size)
        res = _pick_resolution(spec, resolution, notes)
        if res and "resolution" in fields:
            payload["resolution"] = res
    elif geom == "ar_opt":
        if ar_allowed:
            payload["aspect_ratio"] = _pick_aspect_ratio(aspect_ratio, ar_allowed, notes, src_size)
    else:  # none — модель вообще не принимает геометрию
        if w or h:
            notes.append("модель не принимает размеры: width/height/exact_* проигнорированы.")
        return
    # страховка: не отправляем поля, которых нет в схеме
    for bad in ("width", "height", "aspect_ratio", "resolution"):
        if bad in payload and bad not in fields:
            payload.pop(bad, None)


def _ref_field(spec: dict):
    """Какое поле модели используется для референсов: ('image_urls', 10) и т.п."""
    refs = spec.get("refs") or {}
    for f in REF_FIELD_ORDER:
        if f in refs:
            return f, int(refs[f] or DEFAULT_MAX_REFS)
    return None, 0


def _pack_refs(field: str, media: list, notes: list) -> list:
    """Упаковать референсы в формате конкретного поля (у каждого своя схема)."""
    if field in ("style_images", "image_style_references"):
        return [{"url": u, "strength": 0.8} for u in media]
    if field == "reference_images":  # runway: объекты {url, tag}
        return [{"url": u, "tag": f"ref{i + 1}"} for i, u in enumerate(media)]
    return list(media)  # image_urls / character_reference_images — просто строки


# ---------------------------------------------------------------------------
# Локальный реестр задач (страховка от «потеряли job_id»)
# ---------------------------------------------------------------------------

_REG_LOCK = threading.Lock()
_REG_JOBS = []          # newest first


def _reg_add(entry: dict, state_file: str = "") -> None:
    with _REG_LOCK:
        _REG_JOBS.insert(0, entry)
        del _REG_JOBS[REGISTRY_LIMIT:]
        snapshot = list(_REG_JOBS)
    if state_file:
        try:
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False)
        except Exception:
            pass  # реестр — вспомогательный: его недоступность не должна ломать генерацию


def _reg_all(state_file: str = "") -> list:
    with _REG_LOCK:
        if _REG_JOBS:
            return list(_REG_JOBS)
    if state_file and os.path.exists(state_file):
        try:
            with open(state_file, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                with _REG_LOCK:
                    if not _REG_JOBS:
                        _REG_JOBS.extend(data[:REGISTRY_LIMIT])
                return list(data[:REGISTRY_LIMIT])
        except Exception:
            pass
    return []


def _reg_resolve(token: str, state_file: str = "") -> str:
    """'last', 'last:2', '3' → job_id из реестра (или '')."""
    t = (token or "").strip().lower()
    if not t:
        return ""
    idx = 0
    m = re.fullmatch(r"(?:last)?[:\-\s]*(\d{1,2})", t)
    if t in ("last", "последняя", "last_job"):
        idx = 0
    elif m:
        idx = max(0, int(m.group(1)) - 1) if ":" in t or t.isdigit() else 0
    else:
        return ""
    jobs = _reg_all(state_file)
    return jobs[idx].get("job_id", "") if len(jobs) > idx else ""


# ===========================================================================
#                                  TOOLS
# ===========================================================================

class Tools:
    def __init__(self):
        self.valves = self.Valves()
        # состояние процесса: LRU-кэш добытых указателей и карта «URL → размер»
        self._cache = OrderedDict()        # key -> (pointer, (w, h) | None)
        self._cache_lock = threading.Lock()
        self._size_by_pointer = {}
        self._last_resolve_warning = ""
        # Общие инструкции вставляются в docstring'и методов: Open WebUI строит
        # JSON-схему инструмента из экземпляра Tools(), поэтому подстановка в
        # __init__ попадает и в схему, и в текст, который читает модель.
        for name in ("upload_asset", "generate_image", "edit_image", "enhance_image",
                     "inspect_image_sources", "krea_status", "list_models",
                     "get_job", "find_jobs", "cancel_job"):
            fn = getattr(type(self), name, None)
            if fn is not None and fn.__doc__:
                doc = fn.__doc__
                if "[FILE_SOURCES]" in doc:
                    doc = doc.replace("[FILE_SOURCES]", FILE_SOURCES)
                if "[AGENT_NOTES]" in doc:
                    doc = doc.replace("[AGENT_NOTES]", AGENT_NOTES)
                if "[IMAGE_MODEL_SELECTOR]" in doc:
                    doc = doc.replace("[IMAGE_MODEL_SELECTOR]", IMAGE_MODEL_SELECTOR)
                fn.__doc__ = doc

    class Valves(BaseModel):
        api_key: str = Field(
            "",
            description="Krea API Bearer token (krea.ai → аккаунт → API keys). Задаётся в "
                        "Admin Settings → Valves этого инструмента.",
        )
        ow_base_url: str = Field(
            "",
            description="Base URL этого же сервера Open WebUI (например http://openwebui:8080). "
                        "Нужен, чтобы инструмент скачивал байты вложений чата через внутренний API "
                        "/api/v1/files/{id}/content. Пусто = взять base_url текущего запроса.",
        )
        terminal_url: str = Field(
            "",
            description="URL HTTP API контейнера Open Terminal / «Файлового хранилища» "
                        "(например http://open-terminal:8000). Пусто = чтение путей ФС отключено.",
        )
        terminal_api_key: str = Field(
            "",
            description="Bearer-токен API Open Terminal (тот же, что использует MCP-клиент чата). "
                        "Нужен только если заполнен terminal_url.",
        )
        max_media_bytes: int = Field(
            12_000_000,
            description="Потолок размера одной картинки (байт) при скачивании из чата/хранилища/URL. "
                        "Выше — ошибка с подсказкой уменьшить файл. Защита памяти контейнера Open WebUI.",
        )
        media_cache_bytes: int = Field(
            24_000_000,
            description="Бюджет LRU-кэша уже добытых указателей на картинки (байт). Повторные "
                        "ссылки на одну картинку не перекачиваются и не перезаливаются.",
        )
        inline_data_uri_max_bytes: int = Field(
            3_500_000,
            description="Картинки больше этого размера НЕ вставляются в запрос как data URI, а "
                        "заливаются на Krea (POST /assets) с подстановкой короткого URL. Ограничение "
                        "не по схеме Krea (её maxLength=1024 НЕ проверяется — проверка A, 20.09.2026), "
                        "а по размеру тела: data URI ≈1.33 x байт против payload_max_bytes=6 МБ. "
                        "3.5 МБ дают тело ≈4.7 МБ, т.е. ~22% запаса. Меняй только осознанно.",
        )
        payload_max_bytes: int = Field(
            6_000_000,
            description="Потолок размера JSON-тела запроса к Krea. Превышение = отказ с подсказкой "
                        "(обычно значит, что картинка ушла как data URI).",
        )
        wait_s_max: int = Field(
            300,
            description="Жёсткий потолок ожидания результата внутри одного вызова (секунды). "
                        "Большие значения упираются в таймауты фронта Open WebUI.",
        )
        max_active_jobs: int = Field(
            6,
            description="Сколько одновременных незавершённых задач Krea считаем нормой. При превышении "
                        "инструмент откажется создавать новую задачу (защита от 429 и лишних трат).",
        )
        guard_active_jobs: bool = Field(
            True,
            description="Проверять перед отправкой задачи, не перегружен ли аккаунт активными задачами.",
        )
        state_file: str = Field(
            "/tmp/krea_jobs_registry.json",
            description="Файл, куда складывается локальный реестр последних задач (страховка от потери "
                        "job_id). Пусто = только память процесса.",
        )

    # ------------------------------------------------------------- транспорт

    def _headers(self, json_body: bool = True) -> dict:
        if not self.valves.api_key:
            raise KreaApiError("Krea API key не задан: Admin → Valves этого инструмента → api_key.")
        h = {"Authorization": f"Bearer {self.valves.api_key}"}
        if json_body:
            h["Content-Type"] = "application/json"
        return h

    def _timeout(self, read_s: int = None):
        return (HTTP_CONNECT_TIMEOUT_S, read_s or HTTP_READ_TIMEOUT_S)

    def _http_sync(self, method: str, path: str, *, params=None, json_body=None, files=None,
                   read_timeout: int = None, retries: int = 1, headers: dict = None,
                   url: str = None, stream: bool = False) -> requests.Response:
        """Синхронный HTTP-вызов с ретраями. Всегда запускается через _async()."""
        target = url or (BASE_URL + path)
        hdrs = headers if headers is not None else self._headers(json_body is not None)
        attempt = 0
        while True:
            attempt += 1
            try:
                return requests.request(method, target, params=params, json=json_body,
                                        files=files, headers=hdrs, stream=stream,
                                        timeout=self._timeout(read_timeout))
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt >= max(1, retries):
                    raise KreaTransport(f"{type(e).__name__} на {method} {target}: {e}") from e
                time.sleep(RETRY_BACKOFF_S * (2 ** (attempt - 1)))
            except requests.exceptions.RequestException as e:  # прочие сетевые сбои
                raise KreaTransport(f"{type(e).__name__} на {method} {target}: {e}") from e

    async def _async(self, fn, *args, **kwargs):
        """Выполнить блокирующий вызов в отдельном потоке (не морозим event loop OW)."""
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except AttributeError:  # pragma: no cover - очень старый Python
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    def _check_response(self, resp: requests.Response, path: str = "") -> str:
        """Превратить неуспешный ответ в человекочитаемую ошибку (или '' если ok)."""
        if resp.status_code < 400:
            return ""
        body = (resp.text or "")[:400]
        hint = _http_hint(resp.status_code, body)
        extra = ""
        if resp.status_code == 429 and "concurrent" in body.lower():
            extra = " Уменьши частоту: сначала дождись уже запущенных задач."
        return f"{hint}{extra} (HTTP {resp.status_code}{', ' + path if path else ''}): {body}"

    # ------------------------------------------------------- доступ к сервису

    async def _active_jobs(self):
        """(кол-во активных задач, подробности) — бесплатный GET /jobs."""
        try:
            resp = await self._async(self._http_sync, "GET", "/jobs", params={"limit": 50},
                                     retries=RETRY_ATTEMPTS)
        except KreaTransport as e:
            return -1, str(e)
        except KreaApiError as e:
            return -1, str(e)
        if resp.status_code >= 400:
            return -1, self._check_response(resp, "/jobs")
        try:
            items = (resp.json() or {}).get("items") or []
        except Exception:
            return -1, "Krea вернул не-JSON на GET /jobs."
        active = [j for j in items if (j.get("status") in ACTIVE_STATES)]
        return len(active), active

    async def _guard_capacity(self, label: str) -> str:
        """Пустая строка = можно отправлять; иначе текст отказа."""
        if not self.valves.guard_active_jobs:
            return ""
        active, detail = await self._active_jobs()
        if active < 0:
            return ""  # сервис/сеть недоступны — не блокируем, submit сам сообщит ошибку
        limit = max(1, int(self.valves.max_active_jobs or 6))
        if active >= limit:
            lines = [f"⛔ Не отправляю «{label}»: у аккаунта уже {active} незавершённых задач "
                     f"(порог max_active_jobs={limit}). Сначала дождись этих: Krea разбирает "
                     f"очередь примерно по 2 задачи за раз (замер 20.09.2026: при 8 задачах 429 "
                     f"ещё не пришёл, но очередь растянулась на ~19 c), а кредиты за уже "
                     f"запущенные списаны."]
            lines.append("Что делать: вызови find_jobs(status='processing') или get_job для "
                         "уже запущенных задач, дождись результата и только потом запускай новую.")
            for j in (detail or [])[:8]:
                lines.append(f"  - {j.get('job_id')} · {j.get('status')} · {_fmt_dt(j.get('created_at'))}")
            return "\n".join(lines)
        return ""

    async def krea_status(self) -> str:
        """
        ПРОВЕРКА СЕРВИСА (бесплатно, кредиты не тратятся). Вызывай первым, когда:
        генерация падает/висит, вернулись таймауты, «job_id потерян», или просто
        перед большой сессией генераций.

        Проверяет: задан ли API-ключ, жив ли Krea (GET /jobs), его задержку,
        сколько задач уже выполняется (активных), последние задачи и их ошибки,
        доступность мостов к Open WebUI (вложения чата) и к Open Terminal (пути ФС).

        Возвращает JSON с готовым планом действий в поле order_of_actions.

        [AGENT_NOTES]
        """
        out = {
            "krea_api": {"base_url": BASE_URL},
            "api_key": "задан" if self.valves.api_key else "НЕ ЗАДАН (Valves → api_key)",
            "bridge_chat_files": None,
            "bridge_open_terminal": None,
            "active_jobs": None,
            "last_jobs": [],
            "limits": {
                "max_media_bytes": self.valves.max_media_bytes,
                "payload_max_bytes": self.valves.payload_max_bytes,
                "wait_s_max": self.valves.wait_s_max,
                "max_active_jobs": self.valves.max_active_jobs,
                "inline_data_uri_max_bytes": self.valves.inline_data_uri_max_bytes,
            },
        }
        # мост к вложениям чата
        try:
            out["bridge_chat_files"] = (f"{self.valves.ow_base_url.rstrip('/')} + сессия запроса"
                                        if self.valves.ow_base_url else
                                        "ow_base_url не задан — берём base_url текущего запроса; "
                                        "если скачивание вложений падает, задай ow_base_url в Valves")
        except Exception:
            pass
        # мост к файловому хранилищу
        turl = (self.valves.terminal_url or "").strip().rstrip("/")
        if turl and self.valves.terminal_api_key:
            try:
                hr = await self._async(requests.get, f"{turl}/health",
                                       headers={"Authorization": f"Bearer {self.valves.terminal_api_key}"},
                                       timeout=8)
                out["bridge_open_terminal"] = f"{turl} → HTTP {hr.status_code}"
            except Exception as e:
                out["bridge_open_terminal"] = f"{turl} → недоступен: {type(e).__name__}: {e}"
        else:
            out["bridge_open_terminal"] = ("не настроен (Valves → terminal_url + terminal_api_key): "
                                           "пути файлового хранилища не принимаются")

        # сам Krea
        t0 = time.time()
        if not self.valves.api_key:
            out["krea_api"].update({"http": None, "error": "нет ключа"})
        else:
            try:
                resp = await self._async(self._http_sync, "GET", "/jobs", params={"limit": 10},
                                         read_timeout=20, retries=2)
                ms = int((time.time() - t0) * 1000)
                items = []
                try:
                    items = (resp.json() or {}).get("items") or []
                except Exception:
                    items = []
                out["krea_api"].update({
                    "http": resp.status_code,
                    "latency_ms": ms,
                    "verdict": "ok" if resp.status_code < 400 else _http_hint(resp.status_code),
                })
                if resp.status_code >= 400:
                    out["krea_api"]["error"] = (resp.text or "")[:300]
                active = [j for j in items if j.get("status") in ACTIVE_STATES]
                out["active_jobs"] = len(active)
                out["active_job_ids"] = [j.get("job_id") for j in active]
                out["last_jobs"] = [{
                    "job_id": j.get("job_id"),
                    "status": j.get("status"),
                    "type": j.get("type"),
                    "created_at": _fmt_dt(j.get("created_at")),
                    "url": (_extract_urls(j) or [""])[0],
                    "error": _job_error(j) or None,
                } for j in items[:5]]
                if len(active) >= max(1, int(self.valves.max_active_jobs or 6)):
                    out["warning"] = (f"{len(active)} незавершённых задач — новые генерации могут "
                                      f"получить 429. Сначала дождись их (get_job / find_jobs).")
            except KreaTransport as e:
                out["krea_api"].update({"http": None, "latency_ms": int((time.time() - t0) * 1000),
                                        "verdict": f"сеть недоступна из контейнера Open WebUI: {e}"})
        # локальный реестр
        reg = _reg_all(self.valves.state_file)
        out["local_registry"] = [{"job_id": r.get("job_id"), "state": r.get("status"),
                                  "label": r.get("label"), "created_at": _fmt_dt(r.get("created_at"))}
                                 for r in reg[:5]]
        out["order_of_actions"] = [
            "1. Убедись, что krea_api.http == 200 и api_key == «задан». Иначе — Valves/баланс/ключ.",
            "2. Проверь active_jobs: если он близок к limits.max_active_jobs — сначала дождись этих задач.",
            "3. Нужны поля/лимиты/цена модели — вызови list_models(model='...').",
            "4. Генерация: generate_image / edit_image / enhance_image (wait_s не больше "
            f"{self.valves.wait_s_max}).",
            "5. Ответ «ещё выполняется» — это НОРМАЛЬНО: вызови get_job(job_id='...'), не пересоздавай задачу.",
            "6. Потерялся job_id — find_jobs(minutes=30): вернёт id и ссылки по времени создания.",
            "7. Задача уже не нужна — cancel_job(job_id='...') (DELETE /jobs/{id}).",
        ]
        return json.dumps(out, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------- модели

    async def list_models(self, model: str = "") -> str:
        """
        СПРАВОЧНИК МОДЕЛЕЙ (локально, без обращения к сети и без кредитов).
        Вызывай, когда надо понять: какие поля принимает модель, какие обязательны,
        какие aspect_ratio/resolution допустимы, сколько референсов можно передать,
        сколько стоит запрос и какие есть алиасы.

        Без параметра — вернёт список алиасов с ценой. С параметром model — полную
        карточку этой модели (поля, обязательные поля, enum'ы, лимиты, цена).

        :param model: алиас или полный id модели ('seedream-4', 'image/krea/krea-2/large').
            Пусто = сводный список.
        """
        if model.strip():
            try:
                route = _route(model)
            except ValueError as e:
                return f"Error: {e}"
            spec = _spec(route)
            if not spec:
                return f"Error: нет данных по {route}."
            card = {
                "alias": _known(route),
                "route": route,
                "usable_for_generate": "prompt" in (spec.get("required") or []) or "prompt" in (spec.get("fields") or []),
                "required_fields": spec.get("required"),
                "all_fields": spec.get("fields"),
                "geometry": spec.get("geometry"),
                "aspect_ratios": spec.get("aspect_ratios"),
                "resolutions": spec.get("resolutions"),
                "size_bounds_px": spec.get("size_bounds"),
                "quality": spec.get("quality"),
                "max_reference_images": spec.get("refs") or {"image_urls": DEFAULT_MAX_REFS},
                "price_usd": {"min": spec.get("price_min"), "max": spec.get("price_max"),
                              "tiers": spec.get("price_tiers")},
                "notes": {
                    "wh": "размер задаётся width/height",
                    "ar": "размер задаётся aspect_ratio (+resolution обязателен)",
                    "both": "можно width/height или aspect_ratio",
                    "ar_opt": "aspect_ratio необязателен, width/height не принимается",
                    "none": "геометрия не принимается (модель правит уже готовый размер)",
                }.get(spec.get("geometry"), ""),
            }
            return json.dumps(card, ensure_ascii=False, indent=2)
        rows = ["Алиас | route | цена, USD | размер задаётся | макс. рефов"]
        for alias, route in sorted(MODEL_ROUTES.items()):
            spec = _spec(route) or {}
            price = ("—" if spec.get("price_min") is None else
                     (f"{spec['price_min']}" if spec.get("price_min") == spec.get("price_max")
                      else f"{spec['price_min']}–{spec['price_max']}"))
            refs = spec.get("refs") or {}
            mx = max([v for v in refs.values() if isinstance(v, int)] or [0]) or None
            rows.append(f"{alias} | {route} | {price} | {spec.get('geometry', '?')} | {mx or '—'}")
        rows.append("\nЦены — из OpenAPI-спеки Krea (x-krea-pricing): минимальная/максимальная "
                    "для модели (тариф зависит от resolution/quality/кол-ва референсов). "
                    "Полные тарифы: list_models(model='nano-banana-pro').")
        return "\n".join(rows)

    # ------------------------------------------------------------- медиа

    def _cache_get(self, key: str):
        with self._cache_lock:
            item = self._cache.get(key)
            if item is not None:
                self._cache.move_to_end(key)
            return item

    def _cache_put(self, key: str, pointer: str, size=None) -> None:
        with self._cache_lock:
            self._cache[key] = (pointer, size)
            self._cache.move_to_end(key)
            budget = max(0, int(self.valves.media_cache_bytes or 0))
            while sum(len(v[0]) for v in self._cache.values()) > budget and len(self._cache) > 1:
                self._cache.popitem(last=False)

    def _download_capped(self, url: str, *, cap: int, headers: dict = None,
                         cookies: dict = None, read_timeout: int = 120, params=None,
                         what: str = "файл") -> tuple:
        """Скачать с жёстким лимитом байт (не материализуем файл целиком в память, если он велик)."""
        cap = int(cap or 0) or 12_000_000
        resp = requests.request("GET", url, headers=headers, cookies=cookies, params=params,
                                stream=True, timeout=(HTTP_CONNECT_TIMEOUT_S, read_timeout))
        if resp.status_code >= 400:
            raise MediaError(f"{what}: HTTP {resp.status_code} — {(resp.text or '')[:200]}")
        declared = resp.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > cap:
            resp.close()
            raise MediaError(
                f"{what}: {int(declared) / 1e6:.1f} МБ больше лимита "
                f"{cap / 1e6:.1f} МБ (Valves → max_media_bytes). "
                "Уменьши картинку (например до 2048 px по длинной стороне) и повтори.")
        buf = io.BytesIO()
        total = 0
        try:
            for chunk in resp.iter_content(64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > cap:
                    raise MediaError(
                        f"{what}: файл больше лимита {cap / 1e6:.1f} МБ "
                        "(Valves → max_media_bytes). Уменьши картинку и повтори.")
                buf.write(chunk)
        finally:
            resp.close()
        ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        return buf.getvalue(), ctype

    def _ow_base(self, request) -> str:
        base = (self.valves.ow_base_url or "").strip().rstrip("/")
        if not base and request is not None:
            try:
                base = str(request.base_url).rstrip("/")
            except Exception:
                base = ""
        return base

    def _fetch_chat_file_bytes(self, file_id: str, request, files: list) -> bytes:
        """Байты вложения чата через внутренний API Open WebUI (GET /api/v1/files/{id}/content)."""
        file_id = (file_id or "").strip()
        if file_id.lower().startswith("fileid:"):
            file_id = file_id[len("fileid:"):]
        base = self._ow_base(request)
        if not base:
            raise MediaError("Не знаю base URL Open WebUI: заполни Valves → ow_base_url "
                             "(например http://openwebui:8080).")
        headers, cookies = {}, {}
        if request is not None:
            try:
                cookies = dict(getattr(request, "cookies", {}) or {})
            except Exception:
                cookies = {}
            cred = getattr(getattr(getattr(request, "state", None), "token", None), "credentials", None)
            if cred:
                headers["Authorization"] = f"Bearer {cred}"
        if not cookies and not headers:
            raise MediaError(f"Нет сессии пользователя для файла чата {file_id}: инструмент вызван "
                             "вне запроса чата. Передай вместо id путь или https-URL.")
        data, ctype = self._download_capped(f"{base}/api/v1/files/{file_id}/content",
                                            cap=self.valves.max_media_bytes,
                                            headers=headers, cookies=cookies,
                                            what=f"вложение чата {file_id}")
        if not data or not (_is_image_bytes(data) or ctype.startswith("image/")):
            raise MediaError(f"Файл чата {file_id} скачан, но это не картинка "
                             f"(content-type={ctype or 'текст'}, {len(data)} байт).")
        return data

    def _fetch_terminal_file_bytes(self, path: str) -> bytes:
        """Байты файла из «Файлового хранилища» (Open Terminal): GET {terminal_url}/files/read."""
        turl = (self.valves.terminal_url or "").strip().rstrip("/")
        tkey = (self.valves.terminal_api_key or "").strip()
        if not turl or not tkey:
            raise MediaError(f"Путь '{path}' — из файлового хранилища, но мост не настроен: "
                             "Valves → terminal_url и terminal_api_key.")
        data, ctype = self._download_capped(f"{turl}/files/read", cap=self.valves.max_media_bytes,
                                            headers={"Authorization": f"Bearer {tkey}"},
                                            params={"path": path},
                                            what=f"файл '{path}' из хранилища")
        if not (_is_image_bytes(data) or ctype.startswith("image/")):
            raise MediaError(f"Файл '{path}' прочитан, но это не картинка (content-type={ctype or 'текст'}).")
        return data

    def _upload_asset_bytes(self, data: bytes, name: str = "image.png") -> dict:
        """POST /assets → полный ответ Krea.

        Реальные поля (проверено 20.09.2026, проверка B): {id, image_url, uploaded_at,
        width, height, size_bytes, mime_type, description}. В v2.1 и раньше считалось, что
        width/height не отдаются, — отдаются; в v2.2 размеры берутся из этого ответа
        (`_asset_geometry`), а не из локального заголовка файла.
        """
        mime = _mime_by_magic(data)
        name = name or ("image." + ("jpg" if mime == "image/jpeg" else mime.split("/")[-1]))
        resp = requests.post(BASE_URL + "/assets", headers=self._headers(json_body=False),
                             files={"file": (name, io.BytesIO(data), mime)},
                             timeout=(HTTP_CONNECT_TIMEOUT_S, 180))
        if resp.status_code >= 400:
            raise MediaError(self._check_response(resp, "/assets"))
        try:
            payload = resp.json() or {}
        except Exception:
            raise MediaError(f"/assets вернул не-JSON: {(resp.text or '')[:200]}")
        if not payload.get("image_url"):
            raise MediaError(f"/assets не вернул image_url: {str(payload)[:200]}")
        return payload

    def _pointer_to_bytes(self, value: str, request, files: list, cache_key: str = ""):
        """id чата / путь ФС → (bytes, name). URL и data URI байтов не требуют."""
        v = (value or "").strip()
        if _looks_like_file_id(v):
            data = self._fetch_chat_file_bytes(v, request, files)
            return data, self._chat_filename(v, files)
        # иначе считаем путём в файловом хранилище
        try:
            return self._fetch_terminal_file_bytes(v), v
        except MediaError as e:
            if "/" in v or "\\" in v:
                raise e
            try:
                return self._fetch_chat_file_bytes(v, request, files), ""
            except MediaError:
                raise e

    def _resolve_media(self, value: str, request, files: list) -> str:
        """Указатель (id/путь/URL/data URI) → то, что принимает Krea: короткий URL или маленький data URI."""
        v = (value or "").strip()
        if not v:
            raise MediaError("Пустой указатель на изображение.")
        if v.startswith("http://") or v.startswith("https://"):
            return v
        if v.startswith("data:"):
            # большой data URI лучше превратить в asset URL: короче, экономнее по памяти
            approx = len(v) * 3 // 4
            if approx > int(self.valves.inline_data_uri_max_bytes or 0):
                raw = base64.b64decode(v.split(",", 1)[1] or "")
                if len(raw) > int(self.valves.max_media_bytes or 0):
                    raise MediaError(f"data URI слишком большой ({len(raw) / 1e6:.1f} МБ) — "
                                     "уменьши картинку.")
                asset = self._upload_asset_bytes(raw)
                return asset["image_url"]
            return v
        cache_key = f"{'chat' if _looks_like_file_id(v) else 'fs'}:{v}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached[0]
        data, name = self._pointer_to_bytes(v, request, files)
        if not data:
            raise MediaError(f"Не удалось получить байты картинки по '{v}'.")
        size = _size_from_header(data[:96 * 1024])
        if len(data) > int(self.valves.inline_data_uri_max_bytes or 0):
            try:
                asset = self._upload_asset_bytes(data, name=os.path.basename(name or "") or "image.png")
                pointer = asset["image_url"]
                # v2.2: ответ Krea (width/height) точнее локального разбора заголовка
                size = _asset_geometry(asset) or size
            except MediaError as e:
                # залить не удалось (например 402/сеть) — отдаём data URI, но предупреждаем
                pointer = _data_uri(data, name)
                self._last_resolve_warning = (f"не удалось залить '{v}' на Krea ({e}); "
                                              "передаю data URI — при ошибке 400 уменьши картинку")
        else:
            pointer = _data_uri(data, name)
        self._cache_put(cache_key, pointer, size)
        with self._cache_lock:
            if size:
                self._size_by_pointer[pointer] = size
        return pointer

    def _chat_filename(self, file_id: str, files: list, messages: list = None) -> str:
        for item in self._chat_media_items(files, messages):
            if item.get("id") == file_id or item.get("url") == file_id:
                return item.get("name") or ""
        return ""

    @staticmethod
    def _file_id_of(value) -> str:
        """id файла из строки/словаря вложения Open WebUI.

        Понимает как плоский `{id|url|name}`, так и фронтовый `{type, file: {...}}`,
        и url вида `/api/v1/files/<id>/content`.
        """
        if isinstance(value, dict):
            inner = value.get("file") if isinstance(value.get("file"), dict) else {}
            value = value.get("id") or inner.get("id") or value.get("url") or ""
        v = str(value or "").strip()
        if "/files/" in v and v.endswith("/content"):     # url вложения → его id
            v = v.split("/files/")[-1].split("/")[0].split("?")[0]
        return v if _looks_like_file_id(v) else ""

    @classmethod
    def _chat_media_items(cls, files: list, messages: list = None) -> list:
        """Вложения-КАРТИНКИ из __files__, msg['files'] и content-частей image_url.

        Возвращает [{id, name, content_type, url, kind}], где kind:
        file — обычный файл вложения, content — картинка из content-части OW,
        data_uri — картинка, пришедшая как data URI (id файла нет).

        ПОЧЕМУ НЕ ТОЛЬКО __files__ (выяснено 20.09.2026 на Open WebUI 0.11.3):
        фронтенд намеренно выкидывает картинки из payload.files — в `submitPrompt`
        и в `sendMessage` стоит фильтр `item.type === 'file' &&
        !(item?.content_type ?? '').startsWith('image/')`. Бэкенд же делает
        `metadata['files'] = files` и отдаёт это инструментам как `__files__`.
        Итог: у картинок `__files__` ВСЕГДА пуст.

        ПОЧЕМУ И В __messages__ ПУСТО (нашла боевая проверка 20.09.2026, v2.3):
        бэкенд OW сам вырезает картинки из истории перед вызовом инструмента —
        `middleware.py` v0.11.3 ~2431–2453 переносит их в `content` частями
        `{'type': 'image_url', 'image_url': {'url': …}}` и сразу делает
        `message.pop('files', None)`; инструменту отдаётся уже этот
        `form_data['messages']` (строки ~2923 / ~3206 / ~5702). Поэтому читать один
        `msg['files']` бесполезно — картинка видна только в content.
        """
        out, seen = [], set()

        def push(fid, name, ctype, url="", kind="file"):
            ctype = str(ctype or "")
            if ctype and not ctype.startswith("image/"):
                return
            if fid and not ctype:
                ctype = "image/*"      # id файла есть — значит это картинка вложения
            key = fid or url
            if not key or key in seen:
                return
            seen.add(key)
            out.append({"id": fid or "", "name": name or "", "content_type": ctype,
                        "url": url, "kind": kind})

        for f in files or []:
            if not isinstance(f, dict):
                continue
            meta = f.get("meta") if isinstance(f.get("meta"), dict) else {}
            ctype = f.get("content_type") or meta.get("content_type")
            url = str(f.get("url") or "")
            fid = cls._file_id_of(f)
            if not fid and url.startswith(("http://", "https://")):
                push("", f.get("name") or f.get("filename"), ctype, url)   # напр. Google Drive
                continue
            push(fid, f.get("name") or f.get("filename"), ctype, url)

        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            for f in msg.get("files") or []:
                if not isinstance(f, dict):
                    continue
                meta = f.get("meta") if isinstance(f.get("meta"), dict) else {}
                push(cls._file_id_of(f), f.get("name") or f.get("filename"),
                     f.get("content_type") or meta.get("content_type"), str(f.get("url") or ""))
            # v2.3: картинки, перенесённые бэкендом OW из files в content
            content = msg.get("content")
            if isinstance(content, str) and 0 < len(content) <= 200_000:
                for hit in _CONTENT_MEDIA_RE.finditer(content):
                    raw = hit.group(0)
                    if raw.lower().startswith("data:image/"):
                        push("", "вложение (data URI в тексте)", "image/*", raw, kind="data_uri")
                    else:
                        push(cls._file_id_of(raw), "вложение (ссылка в тексте)", "image/*", raw,
                             kind="content")
            for part in (content if isinstance(content, list) else []):
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                iu = part.get("image_url")
                url = (iu.get("url") if isinstance(iu, dict) else iu) if iu else ""
                url = str(url or "").strip()
                if not url:
                    continue
                name = str(part.get("name") or "")
                ctype = part.get("content_type") or "image/*"
                if url.startswith("data:"):
                    push("", name or "вложение (data URI)", ctype, url, kind="data_uri")
                    continue
                push(cls._file_id_of(url), name, ctype, url, kind="content")
        return out

    def _first_chat_media(self, files: list, request, messages: list = None) -> str:
        items = self._chat_media_items(files, messages)
        if not items:
            return ""
        # у картинок из content бывает либо id файла, либо url — берём то, что есть
        pointer = str(items[0].get("id") or items[0].get("url") or "")
        return self._resolve_media(pointer, request, files or [])

    def _media_or_none(self, value: str, request, files: list, messages: list = None) -> str:
        if (value or "").strip():
            return self._resolve_media(value, request, files)
        return self._first_chat_media(files, request, messages)

    def _media_list(self, values: list, request, files: list) -> list:
        out, seen = [], set()
        for v in values or []:
            if isinstance(v, str) and v.strip():
                p = self._resolve_media(v, request, files)
                if p and p not in seen:   # дубликаты не гоняем дважды
                    seen.add(p)
                    out.append(p)
        return out

    def _media_size_of(self, media: str):
        """Размер по data URI или по уже загруженному asset URL (из кэша)."""
        size = _media_size(media)
        if size:
            return size
        with self._cache_lock:
            return self._size_by_pointer.get(media)

    def _probe_size(self, url: str):
        """Лучшие усилия: узнать размер удалённой картинки, скачав только первые 64 КБ."""
        try:
            r = requests.get(url, headers={"Range": "bytes=0-65535"},
                             timeout=(HTTP_CONNECT_TIMEOUT_S, 15), stream=True)
            data = r.raw.read(96 * 1024, decode_content=True) or b""
            r.close()
            return _size_from_header(data)
        except Exception:
            return None

    # ------------------------------------------------------------ jobs

    def _iso_now_minus(self, seconds: int) -> str:
        from datetime import datetime, timedelta, timezone
        return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")

    async def _recover_lost_job(self, t0: float, label: str) -> str:
        """После сетевого сбоя submit: найти задачу, которая могла создаться."""
        try:
            resp = await self._async(self._http_sync, "GET", "/jobs", params={"limit": 25},
                                     retries=2, read_timeout=25)
        except (KreaTransport, KreaApiError):
            return ""
        if resp.status_code >= 400:
            return ""
        try:
            items = (resp.json() or {}).get("items") or []
        except Exception:
            items = []
        # кандидаты: всё, что создано не раньше старта отправки минус 10 c (запас на рассинхрон часов)
        candidates = []
        cutoff = t0 - 10
        for j in items:
            created = _parse_dt(j.get("created_at"))
            if created and created.timestamp() >= cutoff:
                candidates.append(j)
        if not candidates:
            return ""
        lines = [f"⚠️ «{label}»: связь с Krea оборвалась на отправке, но задача МОГЛА создаться.",
                 "Нашёл свежие задачи в аккаунте (проверь, какая из них твоя, и не создавай дубликат):"]
        for j in candidates[:5]:
            lines.append(f"  - job_id={j.get('job_id')} · status={j.get('status')} · "
                         f"{_fmt_dt(j.get('created_at'))} → get_job(job_id='{j.get('job_id')}')")
        if len(candidates) == 1:
            jid = candidates[0].get("job_id")
            lines.append(f"Скорее всего это она. Дальше: get_job(job_id='{jid}').")
        return "\n".join(lines)

    async def _submit(self, route: str, payload: dict, label: str) -> tuple:
        """POST /generate/{route}. Возвращает (job_id|'', текст_ошибки|'')."""
        t0 = time.time()
        try:
            resp = await self._async(self._http_sync, "POST", f"/generate/{route}", json_body=payload,
                                     read_timeout=SUBMIT_READ_TIMEOUT_S, retries=1)
        except KreaApiError as e:
            return "", f"❌ «{label}»: {e}"
        except KreaTransport as e:
            return "", (await self._recover_lost_job(t0, label) or
                        f"⚠️ «{label}»: запрос не дошёл до Krea ({e}).\n"
                        "Задача могла создаться — вызови find_jobs(minutes=10) и проверь, "
                        "прежде чем повторять (повтор = ещё одна оплата).")
        if resp.status_code >= 400:
            return "", f"❌ «{label}»: {self._check_response(resp, route)}"
        try:
            job = resp.json() or {}
        except Exception:
            return "", f"❌ «{label}»: Krea вернул не-JSON (HTTP {resp.status_code}): {(resp.text or '')[:300]}"
        job_id = job.get("job_id") or ""
        if not job_id:
            return "", (f"❌ «{label}»: в ответе нет job_id — {(str(job)[:300])}. "
                        "Проверь find_jobs(minutes=10): задача могла создаться.")
        return job_id, ""

    async def _poll(self, job_id: str, wait_s: int, emitter=None) -> tuple:
        """Опрос задачи. Возвращает (job|None, текст_таймаута|''). Никогда не роняет job_id."""
        deadline = time.monotonic() + max(5, int(wait_s))
        interval = POLL_INTERVAL_S
        errors = 0
        last_status = "?"
        last_job = None
        t_start = time.monotonic()
        while True:
            try:
                resp = await self._async(self._http_sync, "GET", f"/jobs/{job_id}", retries=1,
                                         read_timeout=30)
                if resp.status_code >= 400:
                    if resp.status_code == 404:
                        return None, (f"❌ Krea не знает задачу {job_id} (404). Возможно, задача "
                                      "удалена/устарела — используй find_jobs.")
                    errors += 1
                    if errors >= MAX_POLL_ERRORS:
                        return None, (f"⚠️ Опрос {job_id} сломался: "
                                      f"{self._check_response(resp, f'/jobs/{job_id}')}")
                else:
                    errors = 0
                    try:
                        last_job = resp.json() or {}
                    except Exception:
                        last_job = None
                    if last_job is not None:
                        last_status = last_job.get("status") or "?"
                        if last_status in TERMINAL_STATES:
                            return last_job, ""
                        if last_status == "intermediate-complete" and _extract_urls(last_job):
                            return last_job, ""
            except KreaTransport as e:
                errors += 1
                if errors >= MAX_POLL_ERRORS:
                    return None, (f"⚠️ Опрос {job_id} прерван сетью ({e}). Задача на Krea продолжает "
                                  f"считаться — вызови get_job(job_id=\"{job_id}\") позже.")
            left = deadline - time.monotonic()
            if left <= 0:
                return None, ""
            if emitter and (time.monotonic() - t_start) > 10:
                await self._emit(emitter, f"Krea: {last_status} ({int(time.monotonic() - t_start)} c)…")
            await asyncio.sleep(min(interval, max(0.2, left)))
            interval = min(POLL_MAX_INTERVAL_S, interval * 1.5)

    async def _emit(self, emitter, description: str, done: bool = False) -> None:
        if not emitter:
            return
        try:
            await emitter({"type": "status", "data": {"description": description, "done": done}})
        except Exception:
            pass

    def _sanitize_payload(self, route: str, payload: dict, notes: list) -> list:
        """Убрать поля, которых нет в схеме модели, и вернуть список обязательных пропущенных.

        У схем Krea additionalProperties=false: лишнее поле = HTTP 400, поэтому это
        последний барьер перед отправкой.
        """
        spec = _spec(route)
        allowed = set(spec.get("fields") or ())
        if allowed:
            dropped = [k for k in list(payload) if k not in allowed]
            for k in dropped:
                payload.pop(k, None)
            if dropped:
                notes.append(f"поля {', '.join(dropped)} модель {_known(route)} не принимает — "
                             "не отправлялись (иначе Krea вернул бы 400).")
        return [f for f in (spec.get("required") or []) if f not in payload]

    async def _run_job(self, route: str, payload: dict, wait_s: int, label: str,
                       emitter=None, notes: list = None, prompt: str = "") -> str:
        """Отправить задачу, дождаться (в пределах политики) и вернуть понятный текст."""
        notes = list(notes or [])
        # 0) защита от гигантского тела (обычно data URI в payload) — до любых преобразований
        try:
            body_size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        except Exception:
            body_size = 0
        cap = int(self.valves.payload_max_bytes or 0)
        if cap and body_size > cap:
            return (f"Error: тело запроса {body_size / 1e6:.1f} МБ больше лимита "
                    f"{cap / 1e6:.1f} МБ (Valves → payload_max_bytes).\n"
                    "Почти всегда причина — картинка в payload как data URI. Инструмент сам заливает "
                    "большие картинки на Krea и подставляет короткий URL; если это не сработало — "
                    "уменьши размер файла (например до 2048 px по длинной стороне).")
        # 1) схема: только разрешённые поля, все обязательные на месте
        missing = self._sanitize_payload(route, payload, notes)
        if missing:
            return (f"Error: модель {_known(route)} требует поля {', '.join(missing)}, а их нет. "
                    f"Проверь карточку модели: list_models(model=\"{_known(route)}\").")
        # 2) защита от перегрузки аккаунта (429 = провал + потерянные кредиты)
        refuse = await self._guard_capacity(label)
        if refuse:
            return refuse
        # 3) отправка
        alias = _known(route)
        await self._emit(emitter, f"Krea: отправляю задачу ({alias})…")
        job_id, err = await self._submit(route, payload, label)
        if err:
            return err
        _reg_add({"job_id": job_id, "route": route, "alias": alias, "label": label,
                  "created_at": self._iso_now_minus(0),
                  "prompt": (prompt or "")[:120]}, self.valves.state_file)
        # 4) ожидание
        wait_s = max(5, min(int(wait_s or 0) or 180, int(self.valves.wait_s_max or MAX_WAIT_S)))
        t0 = time.monotonic()
        job, interrupted = await self._poll(job_id, wait_s, emitter)
        elapsed = time.monotonic() - t0
        note_block = ("\n\nПримечания:\n- " + "\n- ".join(notes)) if notes else ""
        if job is None:
            if interrupted:
                return interrupted + f"\njob_id: {job_id}" + note_block
            return (f"⏳ «{label}» ещё выполняется ({elapsed:.0f} c из {wait_s}). Это нормально: задача "
                    f"считается на стороне Krea.\n"
                    f"job_id: {job_id}\n"
                    f"Что делать: НЕ пересоздавай задачу (это ещё один платный запрос) — вызови "
                    f"get_job(job_id=\"{job_id}\") через минуту-другую." + note_block)
        status = job.get("status")
        urls = _extract_urls(job)
        await self._emit(emitter, f"Krea: {status}", done=True)
        if status == "completed":
            if not urls:
                return (f"✅ «{label}»: completed, но Krea не вернул ссылок.\njob_id: {job_id}"
                        + note_block)
            head = (f"✅ «{label}» готово за {elapsed:.0f} c (модель {alias})."
                    if status == "completed" else f"✅ «{label}»: {status} за {elapsed:.0f} c.")
            if status == "intermediate-complete":
                head = f"✅ «{label}»: промежуточный результат за {elapsed:.0f} c."
            return (head + "\n" + "\n".join(urls) +
                    f"\njob_id: {job_id}" + note_block)
        if status in ("failed", "cancelled"):
            return (f"❌ «{label}»: {status}. Причина: {_job_error(job) or 'не указана'}.\n"
                    f"job_id: {job_id}\n"
                    "Что делать: если причина похожа на перегрузку/таймаут — повтори позже; "
                    "если на контент — измени промпт. Повторный запуск = новая оплата." + note_block)
        return (f"⏳ «{label}» ещё выполняется (status={status}, {elapsed:.0f} c).\n"
                f"job_id: {job_id}\nПроверь позже через get_job." + note_block)

    # ------------------------------------------------------- методы-инструменты

    async def upload_asset(self, image: str = "", __files__: list = None,
                           __messages__: list = None,
                           __request__: object = None) -> str:
        """
        Загружает изображение на Krea (POST /assets) и возвращает asset URL — короткий и
        переиспользуемый, который можно много раз подставлять в
        image/image_refs. Нужен, когда одну картинку предстоит использовать в нескольких
        запросах: тогда она заливается один раз. Размеры и вес берутся ИЗ ОТВЕТА Krea
        (width/height/size_bytes) — v2.2.

        [FILE_SOURCES]
        Если image пуст — берётся первое изображение, приложенное к сообщению.

        :param image: указатель на изображение: id файла чата (uuid), абсолютный путь в
            файловом хранилище (Open Terminal), https-URL или data URI.
        :param __files__: вложения сообщения (инжектится Open WebUI — не заполняй сам).
            ВНИМАНИЕ: картинок тут нет — фронтенд OW 0.11.3 отфильтровывает их из
            payload.files (проверено 20.09.2026). Не полагайся на эти параметры:
            надёжнее передать указатель явно.
        :param __messages__: история сообщений чата (инжектится Open WebUI). Поля `files`
            в ней тоже нет — бэкенд вырезает его (message.pop('files')), картинки идут
            content-частями image_url; инструмент их разбирает (v2.3, проверено 20.09.2026).
        :param __request__: контекст запроса (инжектится автоматически).
        """
        try:
            # ОБЯЗАТЕЛЬНО через _async: эта ветка ходит за байтами вложения в сам Open WebUI
            # (GET /api/v1/files/{id}/content). Синхронный requests.get здесь блокирует event
            # loop, сервер не может ответить на собственный запрос — самоблокировка на 120 c
            # (read timeout) и паралич интерфейса OW (проверка E, 20.09.2026).
            media = await self._async(self._media_or_none, image, __request__, __files__,
                                      __messages__)
        except MediaError as e:
            return f"Error: {e}"
        if not media:
            return ("Error: не передано изображение и к сообщению ничего не приложено. "
                    "Передай id файла чата, путь из файлового хранилища или https-URL.")
        try:
            if media.startswith("data:"):
                raw = base64.b64decode(media.split(",", 1)[1])
                size = _size_from_header(raw[:96 * 1024])
                name = "image." + _mime_by_magic(raw).split("/")[-1]
            elif _looks_like_asset_url(media):
                size = self._media_size_of(media)
                return (f"Это уже asset URL Krea — заливать повторно не нужно:\n{media}"
                        + (f"\nразмер: {size[0]}x{size[1]}" if size else ""))
            else:
                raw, _ctype = await self._async(self._download_capped, media,
                                                cap=self.valves.max_media_bytes,
                                                read_timeout=120, what="картинка по URL")
                size = _size_from_header(raw[:96 * 1024])
                name = os.path.basename(media.split("?")[0]) or "image.png"
            asset = await self._async(self._upload_asset_bytes, raw, name)
        except MediaError as e:
            return f"Error: {e}"
        except KreaTransport as e:
            return f"Error: сеть недоступна: {e}"
        # v2.2: Krea сама сообщает размеры и вес залитого файла — доверяем ответу
        # (проверка B, 20.09.2026), локальный заголовок остаётся резервом.
        size_from_krea = _asset_geometry(asset)
        bytes_from_krea = _asset_size_bytes(asset)
        if size_from_krea:
            size = size_from_krea
        with self._cache_lock:
            if size:
                self._size_by_pointer[asset["image_url"]] = size
        source = "из ответа Krea" if size_from_krea else "по заголовку файла"
        return (f"Загружено (asset id={asset.get('id')}). Asset URL:\n{asset['image_url']}\n"
                + (f"размер: {size[0]}x{size[1]} ({source}), " if size else "")
                + f"{bytes_from_krea or len(raw)} байт. Дальше передавай этот URL в image/image_refs — он короткий и "
                  "переиспользуемый (повторная загрузка не нужна).")

    async def inspect_image_sources(self, __files__: list = None, __messages__: list = None,
                                    __request__: object = None) -> str:
        """
        ДИАГНОСТИКА ДОСТУПНЫХ КАРТИНОК: список вложений текущего чата (id, имя, тип) и
        состояние мостов (Open WebUI / Open Terminal). Вызывай, если непонятно, какой
        указатель передать в image/image_refs, или если предыдущая передача файла упала.

        [FILE_SOURCES]

        :param __files__: файлы сообщения (инжектится автоматически). Картинки сюда
            НЕ попадают: Open WebUI 0.11.3 фильтрует их из payload.files (проверено
            20.09.2026).
        :param __messages__: история сообщений чата (инжектится автоматически). Поля
            `files` в ней тоже нет (бэкенд вырезает: message.pop('files')), картинки
            приходят content-частями image_url — именно их и показывает chat_files.
        :param __request__: контекст запроса (инжектится автоматически).
        """
        items = self._chat_media_items(__files__, __messages__)
        shown = []
        for it in items:
            ptr = str(it.get("id") or it.get("url") or "")
            if ptr.startswith("data:"):
                # целый data URI в ответ печатать нельзя (мегабайты) — только признаки
                shown.append({"id": "", "name": it.get("name") or "", "kind": it.get("kind"),
                              "content_type": it.get("content_type") or "",
                              "pass_as": f'image="<data URI {len(ptr)} симв. — копировать не нужно: '
                                         'передавай явный id / путь / URL>"'})
                continue
            entry = {"id": it.get("id") or "", "name": it.get("name") or "",
                     "kind": it.get("kind"), "content_type": it.get("content_type") or "",
                     "pass_as": f'image="{ptr}"'}
            url = str(it.get("url") or "")
            if url and url != ptr and len(url) <= 200:
                entry["url"] = url
            shown.append(entry)
        files_in_hist = sum(len(m.get("files") or []) for m in (__messages__ or [])
                            if isinstance(m, dict))
        content_parts = 0
        for m in (__messages__ or []):
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, list):
                content_parts += sum(1 for p in c
                                     if isinstance(p, dict) and p.get("type") == "image_url")
        reasons = []
        if not (__files__ or []):
            reasons.append("__files__ пуст (фронтенд фильтрует image/)")
        if not files_in_hist:
            reasons.append("в истории нет поля files — бэкенд OW вырезает его "
                           "(middleware.py: message.pop('files')), поэтому картинки "
                           "видны только в content")
        note = ("Open WebUI 0.11.3: " + "; ".join(reasons) + ". " if reasons
                else "Источники вложений в порядке. ")
        note += (f"Разобрано content-частей image_url: {content_parts}; найдено вложений: "
                 f"{len(items)} (v2.3).")
        out = {"chat_files": shown, "terminal_bridge": None, "how_to_pass": {},
               "limits": {"max_media_bytes": self.valves.max_media_bytes,
                          "inline_data_uri_max_bytes": self.valves.inline_data_uri_max_bytes},
               "messages_seen": len(__messages__ or []),
               "content_image_parts": content_parts,
               "files_param_note": note}
        turl = (self.valves.terminal_url or "").strip().rstrip("/")
        if turl and self.valves.terminal_api_key:
            try:
                hr = await self._async(requests.get, f"{turl}/health",
                                       headers={"Authorization": f"Bearer {self.valves.terminal_api_key}"},
                                       timeout=8)
                out["terminal_bridge"] = f"{turl} → HTTP {hr.status_code} (пути ФС принимаются)"
            except Exception as e:
                out["terminal_bridge"] = f"{turl} → недоступен: {e}"
        else:
            out["terminal_bridge"] = ("не настроен (Valves: terminal_url + terminal_api_key) — "
                                      "пути ФС не принимаются")
        out["how_to_pass"] = {
            "chat_attachment": 'image="<id из chat_files выше>"',
            "filesystem": 'image="<абсолютный путь из list_files хранилища>"',
            "web": 'image="https://..."',
        }
        return json.dumps(out, ensure_ascii=False, indent=2)

    # ------------------------------------------------------ генерация

    async def generate_image(
        self,
        prompt: str,
        model: str = "seedream-4",
        aspect_ratio: str = "1:1",
        resolution: str = "1K",
        negative_prompt: str = "",
        image_refs: list = None,
        exact_width: int = 0,
        exact_height: int = 0,
        seed: int = 0,
        wait_s: int = 180,
        __files__: list = None,
        __request__: object = None,
        __event_emitter__: object = None,
    ) -> str:
        """
        СОЗДАТЬ НОВОЕ ИЗОБРАЖЕНИЕ ИЗ ТЕКСТА (text→image).

        Вызывай, когда просят нарисовать/сгенерировать картинку с нуля («сгенерируй закат»,
        «нарисуй постер», «сделай логотип»), когда править готовый файл не нужно
        (для правок — edit_image, для апскейла — enhance_image).
        Коллаж из нескольких фото пользователя — тоже сюда, с model='nano-banana-pro'
        (или 'gpt-image-2') и референсами в image_refs.

        [FILE_SOURCES]

        [IMAGE_MODEL_SELECTOR]

        [AGENT_NOTES]

        :param prompt: подробное описание картинки на английском: сюжет, стиль, освещение,
            композиция. Например: "A serene ocean sunset, golden light, gentle waves,
            photorealistic". Перепиши формулировку пользователя, не отправляй её сырой.
        :param model: модель (см. «Выбор модели»). По умолчанию 'seedream-4'.
        :param aspect_ratio: пропорции: 1:1 (дефолт), 16:9, 9:16, 4:3, 3:2, 4:5, 2:3, 2:1, 1:2.
            «Горизонтальное» = 16:9, «вертикальное» = 9:16 или 4:5. Если у модели уже/уже
            список (например krea-2, z-image, grok) — инструмент подберёт ближайшую
            разрешённую сам и предупредит об этом в ответе.
        :param resolution: '1K' (дефолт), '2K' или '4K'. Поддерживают nano-banana-2/-pro и
            gpt-image-2; у krea-2 и z-image жёстко только 1K; остальные игнорируют.
        :param negative_prompt: что должно ОТСУТСТВОВАТЬ (watermark, text, blur).
            Поддерживает только qwen-2512.
        :param image_refs: до 10 референсов (у некоторых моделей лимит меньше — см.
            list_models). Каждый элемент — указатель: id файла чата, абсолютный путь в
            файловом хранилище, https-URL, data URI или asset URL от upload_asset.
        :param exact_width: точная ширина, px. Только если пользователь назвал размеры; иначе 0.
        :param exact_height: точная высота, px. Только если пользователь назвал размеры; иначе 0.
        :param seed: фиксированный seed для повторяемости. 0 = случайный.
        :param wait_s: сколько секунд ждать результат внутри вызова (по умолчанию 180,
            потолок задаётся Valves → wait_s_max, не больше 300). Не дождались — вернётся
            job_id, добей методом get_job.
        :param __files__: файлы чата (инжектится автоматически; в text→image вложения
            автоматически НЕ подставляются — референсы передавай явно в image_refs).
        :param __request__: контекст запроса (инжектится автоматически).
        :param __event_emitter__: прогресс в интерфейс (инжектится автоматически).
        """
        notes: list = []
        try:
            route = _route(model)
        except ValueError as e:
            return f"Error: {e}"
        spec = _spec(route)
        if "prompt" not in (spec.get("fields") or ["prompt"]):
            return (f"Error: модель {_known(route)} не принимает текстовый промпт — это модель "
                    "правки/апскейла. Используй edit_image или enhance_image.")
        try:
            media = await self._async(self._media_list, image_refs, __request__, __files__)
        except MediaError as e:
            return f"Error: {e}"
        except KreaTransport as e:
            return f"Error: сеть недоступна при подготовке картинок: {e}"
        warn = getattr(self, "_last_resolve_warning", "")
        if warn:
            notes.append(warn)
            self._last_resolve_warning = ""
        fields = set(spec.get("fields") or ())
        payload: dict = {"prompt": prompt}
        _apply_geometry(spec, payload, aspect_ratio, int(exact_width or 0), int(exact_height or 0),
                        resolution, notes)
        if negative_prompt and "negative_prompt" in fields:
            payload["negative_prompt"] = negative_prompt
        elif negative_prompt:
            notes.append("negative_prompt игнорируется: модель его не поддерживает.")
        if seed and "seed" in fields:
            payload["seed"] = seed
        if media:
            field, mx = _ref_field(spec)
            refs_left = fields - set(payload)
            if field in NON_URL_REF_FIELDS or not field:
                single = "image_url" if "image_url" in fields else None
                if single and media:
                    payload[single] = media[0]
                    if len(media) > 1:
                        notes.append(f"модель принимает только один исходник ({single}): "
                                     "использован первый, остальные отброшены.")
                else:
                    return (f"Error: модель {_known(route)} не принимает референсные изображения. "
                            "Возьми nano-banana-pro, gpt-image-2, seedream-4, ideogram-3, "
                            "krea-2-large или flux-kontext.")
            else:
                if len(media) > mx:
                    notes.append(f"модель принимает максимум {mx} референсов — "
                                 f"лишние ({len(media) - mx}) отброшены.")
                payload[field] = _pack_refs(field, media[:mx], notes)
        if "reference_images" in (spec.get("required") or []) and not payload.get("reference_images"):
            return ("Error: runway-gen-4 требует минимум один референс (reference_images). "
                    "Передай image_refs или возьми другую модель.")
        missing = [f for f in (spec.get("required") or []) if f not in payload]
        if missing:
            hint = ""
            if any(f in ("image_url", "image_urls", "reference_images") for f in missing):
                hint = (" Похоже, нужна не генерация с нуля: для правки готовой картинки — "
                        "edit_image, для апскейла — enhance_image (или передай референсы в image_refs).")
            return (f"Error: модель {_known(route)} требует поля {', '.join(missing)}, а они не "
                    f"заполнены.{hint}")
        return await self._run_job(route, payload, wait_s, "Генерация", __event_emitter__,
                                   notes, prompt=prompt)

    # --------------------------------------------------------- редактирование

    async def edit_image(
        self,
        prompt: str,
        image: str = "",
        additional_images: list = None,
        model: str = "seededit",
        strength: float = 0.0,
        wait_s: int = 180,
        __files__: list = None,
        __messages__: list = None,
        __request__: object = None,
        __event_emitter__: object = None,
    ) -> str:
        """
        РЕДАКТИРОВАТЬ ГОТОВУЮ КАРТИНКУ ПО ТЕКСТОВОЙ ИНСТРУКЦИИ (image+text→image).

        Вызывай, когда пользователь ссылается на существующий файл/фото и просит ИЗМЕНИТЬ
        его («замени кошку на ворону», «убери фон», «сделай белый фон карточки товара»,
        «перенеси персонажа на закат», «отретушируй»). Именно этот метод — правильный,
        генерировать «с нуля» не нужно, даже если правки выглядят большими.
        Отдельный вызов upload_asset НЕ нужен: передай указатель прямо в image.

        [FILE_SOURCES]

        [AGENT_NOTES]

        :param prompt: ИНСТРУКЦИЯ на английском, что изменить: глагол + объект + сохраняемый
            контекст. Пример: "Replace the cat with a black raven sitting in the same spot,
            photorealistic, keep the background and lighting unchanged".
        :param image: исходная картинка — указатель: id файла чата (uuid), абсолютный путь
            в файловом хранилище, https-URL, data URI или asset URL. Пусто = первое
            изображение, приложенное к сообщению.
        :param additional_images: доп. референсы (1–3) в том же формате. Поддерживают
            nano-banana-*, gpt-image-*, grok-edit, flux-kontext (как style) и другие —
            точный максимум смотри в list_models(model=...).
        :param model: 'seededit' (дефолт: замена объектов, ретушь), 'flux-kontext' (сильная
            контекстная правка), 'nano-banana-pro' (умные правки, комбинирование фото),
            'gpt-image-2', 'grok-edit'.
        :param strength: 0.0–1.0, сила изменения (актуально для flux-kontext: 0.4 — мягко,
            1.0 — полностью по инструкции).
        :param wait_s: секунды ожидания (как в generate_image).
        :param __files__: файлы чата (инжектится автоматически; если image пуст — берётся
            первое приложенное изображение).
        :param __messages__: история сообщений — оттуда берутся картинки-вложения:
            в Open WebUI 0.11.3 из __files__ они вырезаны, а в __messages__ лежат
            content-частями image_url (разбираются с v2.3, проверено 20.09.2026).
        :param __request__: контекст запроса (инжектится автоматически).
        :param __event_emitter__: прогресс в интерфейс (инжектится автоматически).
        """
        notes: list = []
        try:
            media = await self._async(self._media_or_none, image, __request__, __files__,
                                      __messages__)
        except MediaError as e:
            return f"Error: {e}"
        except KreaTransport as e:
            return f"Error: сеть недоступна при подготовке картинки: {e}"
        warn = getattr(self, "_last_resolve_warning", "")
        if warn:
            notes.append(warn)
            self._last_resolve_warning = ""
        if not media:
            return ("Error: не передана исходная картинка. Передай в image id файла чата, "
                    "путь из файлового хранилища, https-URL или asset URL — либо приложи "
                    "картинку к сообщению.")
        try:
            route = _route(model)
        except ValueError as e:
            return f"Error: {e}"
        spec = _spec(route)
        fields = set(spec.get("fields") or ())
        try:
            extra = await self._async(self._media_list, additional_images, __request__, __files__)
        except MediaError as e:
            return f"Error: {e}"
        payload: dict = {"prompt": prompt}
        field, mx = _ref_field(spec)
        if "image_urls" in fields:
            refs = [media] + [e for e in extra if e != media]
            if len(refs) > mx:
                notes.append(f"модель принимает максимум {mx} картинок в image_urls — "
                             f"лишние отброшены.")
            payload["image_urls"] = refs[:mx]
        elif "image_url" in fields:
            payload["image_url"] = media
            if extra:
                if "style_images" in fields:
                    payload["style_images"] = _pack_refs("style_images", extra[:mx], notes)
                elif "image_style_references" in fields:
                    payload["image_style_references"] = _pack_refs("image_style_references", extra[:mx], notes)
                else:
                    notes.append(f"модель {_known(route)} не принимает доп. референсы — "
                                 f"{len(extra)} лишних отброшено.")
        else:
            return (f"Error: модель {_known(route)} не принимает исходную картинку. Возьми "
                    "seededit, flux-kontext, nano-banana-pro, gpt-image-2 или grok-edit.")
        if strength and "strength" in fields:
            payload["strength"] = max(0.0, min(1.0, float(strength)))
        # геометрия: у krea-2/z-image aspect_ratio+resolution обязательны
        src = self._media_size_of(media)
        if src is None and media.startswith("http") and "krea.ai" not in media:
            src = await self._async(self._probe_size, media)
        if spec.get("geometry") == "ar":
            _apply_geometry(spec, payload, "", 0, 0, "", notes, src_size=src)
        elif spec.get("geometry") == "wh" and "width" not in (spec.get("required") or []):
            pass  # размер наследуется от исходника
        return await self._run_job(route, payload, wait_s, "Редактирование", __event_emitter__,
                                   notes, prompt=prompt)

    # ------------------------------------------------------------- апскейл

    async def enhance_image(
        self,
        image: str = "",
        target_width: int = 0,
        target_height: int = 0,
        scale_factor: float = 2.0,
        prompt: str = "",
        model: str = "krea-enhance",
        ai_strength: float = 0.4,
        wait_s: int = 240,
        __files__: list = None,
        __messages__: list = None,
        __request__: object = None,
        __event_emitter__: object = None,
    ) -> str:
        """
        АПСКЕЙЛ / ПОВЫШЕНИЕ КАЧЕСТВА И ДЕТАЛЕЙ существующей картинки (enhance/upscale).

        Вызывай, когда просят «увеличь фото», «сделай 4K», «повысь разрешение», «добавь
        резкости и деталей», «восстанови качество» — то есть СОХРАНИТЬ содержимое, подняв
        разрешение/детализацию. НЕ вызывай для рисования/правок — это generate_image /
        edit_image. Задача «расширь кадр и апскейл»: сначала edit_image, затем этот метод.

        Апскейл медленный: типично 1–5 минут. Если не уложилось в wait_s — вернётся job_id,
        добей через get_job (не создавай дубликат!). Krea Enhance — $0.0155+, Topaz — дороже.

        [FILE_SOURCES]

        [AGENT_NOTES]

        :param image: исходная картинка — указатель: id файла чата, абсолютный путь в
            файловом хранилище, https-URL, data URI или asset URL. Пусто = первое
            изображение, приложенное к сообщению.
        :param target_width: ЖЕЛАЕМАЯ итоговая ширина, px (например 3840 для 4K).
        :param target_height: желаемая итоговая высота, px. Обычно вместе с target_width.
        :param scale_factor: во сколько раз увеличить (2 = вдвое), если целевые px не заданы.
            krea-enhance: ≥1; topaz-generative: 1–32.
        :param prompt: необязательная подсказка по деталям («sharper fur, cloud details»).
        :param model: 'krea-enhance' (дешёво, креативно, до 8K) или 'topaz-generative'
            (максимум качества и контроля, до 16K).
        :param ai_strength: 0.1–1.0 — насколько смело дорисовывать детали (krea-enhance).
        :param wait_s: секунды ожидания (дефолт 240 — апскейл медленный).
        :param __files__: файлы чата (инжектится автоматически).
        :param __messages__: история сообщений (оттуда берутся картинки-вложения — в
            Open WebUI 0.11.3 из __files__ они вырезаны, лежат content-частями image_url).
        :param __request__: контекст запроса (инжектится автоматически).
        :param __event_emitter__: прогресс в интерфейс (инжектится автоматически).
        """
        notes: list = []
        try:
            media = await self._async(self._media_or_none, image, __request__, __files__,
                                      __messages__)
        except MediaError as e:
            return f"Error: {e}"
        except KreaTransport as e:
            return f"Error: сеть недоступна при подготовке картинки: {e}"
        if not media:
            return ("Error: не передана картинка. Передай в image id файла чата, путь из "
                    "файлового хранилища, https-URL или asset URL — либо приложи картинку к сообщению.")
        try:
            route = _route(model)
        except ValueError as e:
            return f"Error: {e}"
        spec = _spec(route)
        fields = set(spec.get("fields") or ())
        payload: dict = {"image_url": media}
        if "prompt" in fields:
            payload["prompt"] = prompt or ""
        src = self._media_size_of(media)
        if src is None and media.startswith("http") and "krea.ai" not in media:
            src = await self._async(self._probe_size, media)
        bounds = spec.get("size_bounds") or {}
        factor = max(1.0, float(scale_factor or 2.0))
        if target_width and target_height:
            if src:
                factor = max(target_width / src[0], target_height / src[1])
            tw, th = int(target_width), int(target_height)
        elif src:
            tw = int(round(src[0] * factor))
            th = int(round(src[1] * factor))
        else:
            tw = th = 0
        if "topaz" in route:
            if not (tw and th):
                return ("Error: topaz-generative требует целевые width/height, а размер исходной "
                        "картинки определить не удалось. Передай target_width/target_height "
                        "(например 3840x2160).")
            tw, th = _fit_size(tw, th, bounds, notes, label="итоговый ")
            payload["width"], payload["height"] = tw, th
            payload["upscaling_activated"] = factor > 1.01
            payload["image_scaling_factor"] = min(32, max(1, round(factor, 2)))
        else:
            f = max(1.0, round(factor, 2))
            if "image_scaling_factor" in fields:
                payload["image_scaling_factor"] = f
            if "ai_strength" in fields:
                payload["ai_strength"] = max(0.1, min(1.0, float(ai_strength or 0.4)))
            if tw and th:
                notes.append(f"krea-enhance задаёт масштаб множителем (×{f}); итог будет ≈{tw}x{th}.")
        label = "Апскейл"
        return await self._run_job(route, payload, wait_s, label, __event_emitter__, notes,
                                   prompt=prompt)

    # ------------------------------------------------------------- задачи

    async def get_job(self, job_id: str = "last") -> str:
        """
        СТАТУС И РЕЗУЛЬТАТ ЗАДАЧИ Krea по job_id (бесплатно). Вызывай, когда предыдущий
        метод ответил «ещё выполняется ... job_id=...», и просто подставь этот id.

        Понимает алиасы локального реестра: 'last' (последняя отправленная задача),
        'last:2', '3' — номер задачи в списке последних (см. krea_status → local_registry).
        Возвращает статус, ссылки на результат или текст ошибки задачи.

        [AGENT_NOTES]

        :param job_id: UUID задачи, полученный при запуске, либо алиас 'last'/'last:2'.
        """
        jid = (job_id or "").strip().strip('"\'')
        if not JOB_ID_RE.match(jid):
            resolved = _reg_resolve(jid, self.valves.state_file)
            if not resolved:
                return (f"Error: '{job_id}' не похоже на job_id (нужен UUID) и не найдено в "
                        "локальном реестре. Посмотри последние задачи методом find_jobs(minutes=120).")
            jid = resolved
        try:
            resp = await self._async(self._http_sync, "GET", f"/jobs/{jid}", retries=2,
                                     read_timeout=30)
        except KreaTransport as e:
            return (f"⚠️ Krea недоступен ({e}). job_id={jid} не потерян — повтори get_job позже "
                    "или проверь krea_status.")
        if resp.status_code >= 400:
            return f"❌ {self._check_response(resp, f'/jobs/{jid}')}"
        try:
            job = resp.json() or {}
        except Exception:
            return f"Krea вернул не-JSON: {(resp.text or '')[:300]}"
        status = job.get("status")
        urls = _extract_urls(job)
        lines = [f"status: {status}", f"job_id: {jid}",
                 f"создана: {_fmt_dt(job.get('created_at'))}"
                 + (f", завершена: {_fmt_dt(job.get('completed_at'))}" if job.get("completed_at") else "")]
        if urls:
            lines.append("ссылки на результат:")
            lines.extend(urls)
        if status in ("failed", "cancelled"):
            lines.append(f"ошибка: {_job_error(job) or 'не указана'}")
        if status == "completed" and not urls:
            lines.append("Krea не вернул ссылок для завершённой задачи — проверь её в UI Krea.")
        if status not in TERMINAL_STATES:
            lines.append("ещё выполняется: подожди и вызови get_job снова (повторная генерация НЕ нужна).")
        return "\n".join(lines)

    async def find_jobs(self, minutes: int = 120, status: str = "", limit: int = 10,
                        only_local: bool = False) -> str:
        """
        НАЙТИ СВОИ ЗАДАЧИ (бесплатно, GET /jobs). Главное средство, когда «потерялся job_id»:
        возвращает id, статус, тип, время создания/завершения, ссылки и ошибку последних задач.

        Используй: find_jobs(minutes=30) — что было за последние полчаса;
        find_jobs(status="processing") — что сейчас считается; find_jobs(minutes=1440) — за сутки.
        Локальный реестр (последние задачи, отправленные через этот инструмент, с превью
        промпта) показывается в начале ответа — по нему проще понять, какая задача твоя.

        Ограничение Krea API: фильтр по времени на сервере только курсорный, поэтому метод
        берёт свежую страницу и фильтрует по created_at локально (максимум ~1000 записей).

        [AGENT_NOTES]

        :param minutes: за сколько последних минут показать задачи (по умолчанию 120).
        :param status: фильтр статуса: processing, queued, completed, failed, cancelled и т.п.
            Пусто = все.
        :param limit: сколько задач показать (1–50).
        :param only_local: true = не ходить в Krea, показать только локальный реестр.
        """
        out = []
        reg = _reg_all(self.valves.state_file)
        if reg:
            out.append("ЛОКАЛЬНЫЙ РЕЕСТР (отправлено этим инструментом):")
            for r in reg[:min(10, max(1, int(limit)))]:
                out.append(f"  - {r.get('job_id')} · {r.get('status', '?')} · {r.get('label')} · "
                           f"{_fmt_dt(r.get('created_at'))} · \"{(r.get('prompt') or '')[:60]}\"")
            out.append("")
        if only_local:
            return "\n".join(out) if out else "Локальный реестр пуст."
        limit = max(1, min(50, int(limit or 10)))
        try:
            resp = await self._async(self._http_sync, "GET", "/jobs",
                                     params={"limit": min(1000, max(limit * 4, 50))},
                                     retries=RETRY_ATTEMPTS, read_timeout=30)
        except KreaTransport as e:
            return (f"⚠️ Krea недоступен ({e}). " + ("\n".join(out) if out else ""))
        if resp.status_code >= 400:
            return f"❌ {self._check_response(resp, '/jobs')}"
        try:
            items = (resp.json() or {}).get("items") or []
        except Exception:
            return f"Krea вернул не-JSON на GET /jobs: {(resp.text or '')[:300]}"
        from datetime import datetime, timedelta, timezone
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=max(1, int(minutes or 120)))
        shown = 0
        for j in items:
            created = _parse_dt(j.get("created_at"))
            if created and created < cutoff:
                continue
            if status and j.get("status") != status:
                continue
            urls = _extract_urls(j)
            out.append(f"- {j.get('job_id')} · {j.get('status')} · {j.get('type')} · "
                       f"{_fmt_dt(j.get('created_at'))}"
                       + (f" > {_fmt_dt(j.get('completed_at'))}" if j.get("completed_at") else "")
                       + (f"\n    url: {urls[0]}" if urls else "")
                       + (f"\n    error: {_job_error(j)}" if _job_error(j) else ""))
            shown += 1
            if shown >= limit:
                break
        if not shown:
            out.append(f"За последние {minutes} мин задач не найдено"
                       + (f" со статусом '{status}'" if status else "") + ".")
        return "\n".join(out)

    async def cancel_job(self, job_id: str = "last", confirm: bool = False) -> str:
        """
        УДАЛИТЬ/СНЯТЬ ЗАДАЧУ Krea (DELETE /jobs/{id}). Используй, когда задача больше не
        нужна: например, случайно запущена лишняя, или пошли дубликаты, или надо освободить
        место среди одновременных задач (Krea ограничивает их число, иначе 429).

        Метод НЕ возвращает деньги за уже выполненную работу, поэтому сначала убедись, что
        задача действительно не нужна.

        :param job_id: UUID задачи или алиас 'last'/'last:2' из локального реестра.
        :param confirm: защита от случайного вызова — передай true, чтобы реально удалить.
        """
        jid = (job_id or "").strip().strip('"\'')
        if not JOB_ID_RE.match(jid):
            resolved = _reg_resolve(jid, self.valves.state_file)
            if not resolved:
                return f"Error: '{job_id}' не UUID и не найдено в локальном реестре (find_jobs)."
            jid = resolved
        if not confirm:
            return (f"Готов удалить задачу {jid}, но нужен явный подтверждающий флаг: "
                    f"cancel_job(job_id=\"{jid}\", confirm=true). "
                    "Удаление необратимо и не возвращает оплату за выполненную работу.")
        try:
            resp = await self._async(self._http_sync, "DELETE", f"/jobs/{jid}", retries=2,
                                     read_timeout=30)
        except KreaTransport as e:
            return f"⚠️ Krea недоступен ({e}). Задача не удалена. job_id={jid}."
        if resp.status_code >= 400:
            return f"❌ {self._check_response(resp, f'DELETE /jobs/{jid}')}"
        return (f"Задача {jid} удалена на стороне Krea (DELETE /jobs/{jid} → HTTP "
                f"{resp.status_code}). Если картинка была нужна — придётся генерировать заново.")
