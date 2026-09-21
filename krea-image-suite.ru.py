"""
title: Krea Image Suite
author: Open WebUI Admin
author_url: https://api.krea.ai
description: Генерация и обработка изображений через Krea API (text→image, редактирование, апскейл). Модели: Seedream, Qwen, Krea 2, Nano Banana, GPT Image, Flux Kontext, SeedEdit, Krea Enhance, Topaz. Умеет сам доставать байты картинок из чата Open WebUI и из файлового хранилища Open Terminal.
required_open_webui_version: 0.4.0
requirements: requests
version: 2.0.0
licence: MIT
"""

import asyncio
import base64
import io
import json
import re
import time

import requests

from pydantic import BaseModel, Field

BASE_URL = "https://api.krea.ai/"
POLL_INTERVAL_S = 2

MODEL_ROUTES = {
    # короткие alias → полный route Krea (категория/провайдер/модель)
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
    "gpt-image": "image/openai/gpt-image",
    "gpt-image-2": "image/openai/gpt-image-2",
    "flux": "image/bfl/flux-1-dev",
    "flux-1.1-pro": "image/bfl/flux-1.1-pro",
    "flux-1.1-ultra": "image/bfl/flux-1.1-pro-ultra",
    "flux-kontext": "image/bfl/flux-1-kontext-dev",
    "ideogram-3": "image/ideogram/ideogram-3",
    "grok-image": "image/xai/grok-imagine-2",
    "grok-edit": "image/xai/grok-imagine-2-edit",
    "z-image": "image/z-image/z-image",
    "muse-image": "image/meta/muse-image",
    "luma-uni-1": "image/luma/uni-1",
    "runway-gen-4": "image/runway/gen-4-image",
    "krea-enhance": "enhance/krea/enhance",
    "topaz-generative": "enhance/topaz/generative-enhance",
}

# какие поля принимает каждая модель (Krea-схемы имеют additionalProperties:false —
# лишнее поле = HTTP 400). Сверено с api.krea.ai/openapi.json.
MODEL_FIELDS = {
    "image/bytedance/seedream-4":     {"width", "height", "seed", "style_images"},
    "image/bytedance/seedream-5-pro": {"width", "height", "seed", "style_images"},
    "image/bytedance/seedream-5-lite": {"width", "height", "seed", "style_images"},
    "image/bytedance/seededit":       {"image_url", "seed"},
    "image/qwen/2512":                {"width", "height", "seed", "negative_prompt"},
    "image/ideogram/ideogram-3":      {"width", "height", "seed", "style_images", "character_reference_images"},
    "image/google/nano-banana":       {"width", "height", "aspect_ratio", "image_urls"},
    "image/google/nano-banana-2":     {"width", "height", "aspect_ratio", "resolution", "image_urls"},
    "image/google/nano-banana-flash-lite": {"width", "height", "aspect_ratio", "image_urls"},
    "image/google/nano-banana-pro":   {"width", "height", "aspect_ratio", "resolution", "image_urls"},
    "image/openai/gpt-image":         {"width", "height", "quality", "image_urls", "styles"},
    "image/openai/gpt-image-2":       {"width", "height", "aspect_ratio", "resolution", "quality", "image_urls"},
    "image/bfl/flux-1-dev":           {"width", "height", "seed"},
    "image/bfl/flux-1.1-pro":         {"width", "height", "seed"},
    "image/bfl/flux-1.1-pro-ultra":   {"width", "height", "seed", "raw"},
    "image/bfl/flux-1-kontext-dev":   {"width", "height", "seed", "steps", "strength", "guidance_scale", "style_images", "image_url"},
    "image/krea/krea-2/large":        {"aspect_ratio", "resolution", "seed", "strength", "image_url", "image_style_references", "creativity", "intensity", "complexity", "movement"},
    "image/krea/krea-2/medium":       {"aspect_ratio", "resolution", "seed", "strength", "image_url", "image_style_references", "creativity", "intensity", "complexity", "movement"},
    "image/krea/krea-2/medium-turbo": {"aspect_ratio", "resolution", "seed", "strength", "image_url", "image_style_references", "creativity", "intensity", "complexity", "movement"},
    "image/z-image/z-image":          {"aspect_ratio", "resolution", "seed", "denoising_strength", "image_url", "style_images", "skip_prompt_expansion"},
    "image/xai/grok-imagine-2":       {"aspect_ratio", "resolution", "quality", "style_images"},
    "image/xai/grok-imagine-2-edit":  {"image_urls", "quality", "resolution"},
    "image/meta/muse-image":          {"aspect_ratio", "style_images"},
    "image/luma/uni-1":               {"width", "height", "mode", "style", "output_format", "web_search", "style_images"},
    "image/runway/gen-4-image":       {"width", "height", "seed", "reference_images"},
    "enhance/krea/enhance":           {"image_url", "prompt", "image_scaling_factor", "seed", "ai_strength", "clarity_strength", "resemblance_strength", "sharpness", "rescale_color"},
    "enhance/topaz/generative-enhance": {"image_url", "prompt", "width", "height", "seed", "upscaling_activated", "image_scaling_factor", "creativity", "texture", "sharpen", "denoise", "detail", "face_enhancement", "crop_to_fill", "model", "output_format", "subject_detection"},
}

# куда класть референсные картинки у моделей без image_urls
REF_FIELD_FALLBACK = {
    "style_images": {"image/bytedance/seedream-4", "image/bytedance/seedream-5-pro",
                     "image/bytedance/seedream-5-lite", "image/ideogram/ideogram-3",
                     "image/bfl/flux-1-kontext-dev", "image/z-image/z-image",
                     "image/xai/grok-imagine-2", "image/meta/muse-image",
                     "image/luma/uni-1"},
    "image_style_references": {"image/krea/krea-2/large", "image/krea/krea-2/medium",
                               "image/krea/krea-2/medium-turbo"},
    "character_reference_images": {"image/ideogram/ideogram-3"},
    "reference_images": {"image/runway/gen-4-image"},
}

IMAGE_MODEL_SELECTOR = (
    "  Выбор модели (параметр `model`):\n"
    "  - если пользователь не назвал модель — НЕ передавай этот параметр, возьмётся хороший дефолт;\n"
    "  - 'seedream-4'      — фотореализм + надёжный текст на картинке (обычная фотография);\n"
    "  - 'krea-2-large'    — самая эстетичная выразительная картинка (арт, обложки);\n"
    "  - 'qwen-2512'       — дёшево и быстро, среднее качество (черновик);\n"
    "  - 'nano-banana-pro' / 'gpt-image-2' — «умные» модели: сложные промпты, референсные\n"
    "    фото людей/объектов, надписи и композиции (передавай им image_refs);\n"
    "  - 'flux-1.1-ultra', 'ideogram-3', 'grok-image' — под стиль/бренды/скорость.\n"
    "  Допустим и полный Krea id: 'google/nano-banana-pro', 'bfl/flux-1-kontext-dev' и т.п."
)

# ---------------------------------------------------------------------------
# Единая инструкция «КАК ПРАВИЛЬНО ПОЛУЧАТЬ ФАЙЛЫ». Вставляется в docstring
# каждого метода, работающего с изображениями (маркер [FILE_SOURCES]).
# ---------------------------------------------------------------------------
FILE_SOURCES = """КАК ПРАВИЛЬНО ПОЛУЧАТЬ ИЗОБРАЖЕНИЯ (обязательный раздел):
  Krea принимает картинки ТОЛЬКО в виде: https-URL, data URI (data:image/...;base64,...)
  или asset URL от upload_asset. «Путь к файлу» сам по себе для Krea ничего не значит —
  байты картинки сначала нужно добыть из того места, где файл физически лежит.
  Этот инструмент делает добычу байтов САМ, тебе нужно лишь передать ему ссылку-указатель:
  1) КАРТИНКА В СООБЩЕНИИ ПОЛЬЗОВАТЕЛЯ (вложение чата). Ты видишь её как изображение и/или
     упоминание с id файла (блок attached_files / результат list_chat_files).
     ДЕЙСТВИЕ: передай этот id файла КАК ЕСТЬ в параметр image / image_refs
     (пример: image="9f2c...-uuid"). Инструмент сам скачает байты через внутренний API
     Open WebUI (GET /api/v1/files/{id}/content) под сессией пользователя.
     ВАЖНО: встроенный view_file для изображений возвращает ПУСТОЙ текст (у картинок нет
     извлечённого текста) — им байты не получить, поэтому не «читай» картинку, а передай id сюда.
     Если инструмент ответил, что не смог скачать байты по id: убедись, что файл действительно
     приложен к сообщению, и попроси пользователя приложить его ещё раз.
  2) КАРТИНКА ФАЙЛОМ В ФАЙЛОВОЙ СИСТЕМЕ (Open Terminal / «Файловое хранилище»).
     ДЕЙСТВИЕ: передай АБСОЛЮТНЫЙ путь файла (пример: /home/user/photo.png) в параметр
     image / image_refs — инструмент сам прочитает байты через HTTP API файлового хранилища.
     Путь должен быть реальным (из list_files/read_file этого хранилища), не выдуманным.
     Учти: инструменты read_file/view_file отдают тебе изображение лишь как картинку «на
     просмотр» — перередать её байты из своего контекста ты НЕ можешь, поэтому для обработки
     всегда передавай путь/id сюда, а не пытайся «вложить увиденное».
  3) КАРТИНКА УЖЕ В ИНТЕРНЕТЕ ИЛИ УЖЕ ЗАГРУЖЕНА: передай https-URL или asset URL от
     upload_asset КАК ЕСТЬ — ничего скачивать не нужно.
  4) НЕСКОЛЬКО Картинок (коллаж, референсы): список из любых указателей выше в image_refs —
     каждый элемент разрешается по своим правилам (id, путь, URL).
  ЗАПРЕЩЕНО: передавать путь в надежде, что Krea сам его откроет; передавать вывод
  view_file/read_file как «содержимое картинки»; передавать пути контейнера Open Terminal
  куда-либо кроме параметров этого инструмента."""


class MediaError(Exception):
    """Не удалось превратить указатель пользователя в байты/URL для Krea."""


def _route(model: str) -> str:
    """Resolve short alias or full model id into a Krea route 'category/provider/name'."""
    key = (model or "").strip().lstrip("/")
    if key in MODEL_ROUTES:
        return MODEL_ROUTES[key]
    if "/" in key:  # full id like 'google/nano-banana-pro' without category
        return "image/" + key
    raise ValueError(f"Unknown Krea model: {model!r}")


def _known(key: str) -> str:
    """Full route → known alias (or the route itself for unknown ids)."""
    for alias, route in MODEL_ROUTES.items():
        if route == key:
            return alias
    return key


def _is_image_bytes(data: bytes) -> bool:
    return data[:8].startswith(b"\x89PNG\r\n\x1a\n") or data[:3] == b"\xff\xd8\xff" \
        or data[:4] == b"RIFF" and data[8:12] == b"WEBP" or data[:4] in (b"GIF8",) \
        or data[:2] == b"BM" or data[:4] == b"\x00\x00\x01\x00"


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


def _looks_like_file_id(value: str) -> bool:
    """Open WebUI file id — uuid-подобная строка без '/' и '.'."""
    v = (value or "").strip()
    if v.lower().startswith("fileid:"):
        return True
    if "/" in v or "\\" in v or "." in v:
        return False
    return bool(re.fullmatch(r"[0-9a-fA-F\-]{20,64}", v))


def _size_or_ar(route: str, aspect_ratio: str, w: int, h: int, fields: set, payload: dict) -> None:
    """Fit geometry: exact px if given and allowed, else aspect_ratio if supported,
    else default px for models that require width/height."""
    if w and h and "width" in fields:
        payload["width"], payload["height"] = w, h
    elif "aspect_ratio" in fields:
        payload["aspect_ratio"] = aspect_ratio
    elif "width" in fields:  # models with required width/height but no aspect_ratio
        AM = {"1:1": (1024, 1024), "16:9": (1344, 768), "9:16": (768, 1344),
              "4:3": (1152, 896), "3:2": (1216, 832), "4:5": (896, 1152),
              "2:3": (832, 1216), "2:1": (1408, 704), "1:2": (704, 1408),
              "3:4": (896, 1152), "2.35:1": (1536, 640)}
        payload["width"], payload["height"] = AM.get(aspect_ratio, (1024, 1024))
    # resolution обязателен у krea-2/z-image (enum только 1K); у nb/gpt опционален —
    # generate_image сам подставит 2K/4K, если пользователь попросил
    if "resolution" in fields:
        payload["resolution"] = "1K"


def _closest_aspect(w: int, h: int, allowed: list) -> str:
    """Ближайшая разрешённая пропорция из списка Krea (по лог-разнице сторон)."""
    import math
    target = math.log(w / h)
    best, best_d = allowed[0], 1e9
    for a in allowed:
        try:
            aw, ah = (float(x) for x in a.split(":"))
        except Exception:
            continue
        d = abs(math.log(aw / ah) - target)
        if d < best_d:
            best, best_d = a, d
    return best


def _media_size(media: str):
    """(width, height) из data URI без сетевых запросов; None если разобрать не удалось."""
    import struct
    if not media.startswith("data:") or "," not in media:
        return None
    try:
        data = base64.b64decode(media.split(",", 1)[1])
    except Exception:
        return None
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", data[16:24])
            return w, h
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            if data[12:16] == b"VP8X":
                w = 1 + int.from_bytes(data[24:27], "little")
                h = 1 + int.from_bytes(data[27:30], "little")
                return w, h
            if data[12:16] == b"VP8L":
                b = data[21:25]
                w = 1 + (((b[1] & 0x3F) << 8) | b[0])
                h = 1 + (((b[3] & 0x0F) << 10) | (b[2] << 2) | ((b[1] & 0xC0) >> 6))
                return w, h
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


def _extract_urls(job: dict) -> list:
    """result.urls может быть списком строк, списком {type,url} или dict-мапой."""
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


# ---------------------------------------------------------------------- TOOL


class Tools:
    def __init__(self):
        self.valves = self.Valves()
        # вставляем общие инструкции в docstring'и методов (их читает модель):
        # specs генерируются Open WebUI из экземпляра Tools() при импорте/обновлении tools,
        # поэтому подстановка в __init__ попадает и в JSON-схему, и в промпт модели.
        for name in ("upload_asset", "generate_image", "edit_image", "enhance_image",
                     "inspect_image_sources"):
            fn = getattr(type(self), name, None)
            if fn is not None and fn.__doc__:
                doc = fn.__doc__.replace("[FILE_SOURCES]", FILE_SOURCES)
                doc = doc.replace("[IMAGE_MODEL_SELECTOR]", IMAGE_MODEL_SELECTOR)
                fn.__doc__ = doc

    class Valves(BaseModel):
        api_key: str = Field(
            "",
            description="Krea API Bearer token (account → API keys at krea.ai). Stored in Admin Settings → the Tool's Valves.",
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

    # ---------------------------------------------------- helpers

    def _headers(self) -> dict:
        if not self.valves.api_key:
            raise RuntimeError("Krea API key is not set. Admin → the Tool's Valves → api_key.")
        return {"Authorization": f"Bearer {self.valves.api_key}"}

    def _ow_base(self, request) -> str:
        base = (self.valves.ow_base_url or "").strip().rstrip("/")
        if not base and request is not None:
            try:
                base = str(request.base_url).rstrip("/")
            except Exception:
                base = ""
        return base

    def _fetch_chat_file_bytes(self, file_id: str, request, files: list) -> bytes:
        """Скачать БАЙТЫ вложения чата через внутренний API Open WebUI
        (GET /api/v1/files/{id}/content) под сессией текущего пользователя."""
        file_id = (file_id or "").strip()
        if file_id.lower().startswith("fileid:"):
            file_id = file_id[len("fileid:"):]
        base = self._ow_base(request)
        if not base:
            raise MediaError(
                "Не знаю base URL Open WebUI: заполни Valves → ow_base_url "
                "(например http://openwebui:8080), чтобы инструмент мог скачивать вложения чата.")
        headers = {}
        cookies = {}
        if request is not None:
            try:
                cookies = dict(getattr(request, "cookies", {}) or {})
            except Exception:
                cookies = {}
            tok = getattr(getattr(request, "state", None), "token", None)
            cred = getattr(tok, "credentials", None)
            if cred:
                headers["Authorization"] = f"Bearer {cred}"
        if not cookies and not headers:
            raise MediaError(
                f"Нет сессии пользователя для скачивания файла чата {file_id}: "
                "инструмент вызван вне запроса чата. Передай вместо id путь/URL картинки.")
        r = requests.get(f"{base}/api/v1/files/{file_id}/content",
                         headers=headers, cookies=cookies, timeout=180)
        if r.status_code >= 400:
            raise MediaError(
                f"Open WebUI отдал {r.status_code} на скачивание вложения чата {file_id}: "
                f"{r.text[:200]}. Убедись, что файл приложен к сообщению и у пользователя есть доступ.")
        if not r.content or not _is_image_bytes(r.content):
            raise MediaError(
                f"Файл чата {file_id} скачан, но это не картинка "
                f"(content-type={r.headers.get('content-type')}, {len(r.content)} байт). "
                "Для не-изображений этот инструмент не подходит.")
        return r.content

    def _fetch_terminal_file_bytes(self, path: str) -> bytes:
        """Прочитать БАЙТЫ файла из файлового хранилища (контейнер Open Terminal)
        через его HTTP API: GET {terminal_url}/files/read?path=... (Bearer-авторизация)."""
        turl = (self.valves.terminal_url or "").strip().rstrip("/")
        tkey = (self.valves.terminal_api_key or "").strip()
        if not turl or not tkey:
            raise MediaError(
                f"Путь '{path}' указывает на файловое хранилище (Open Terminal), но мост не настроен: "
                "заполни Valves → terminal_url и terminal_api_key. Либо передай картинку вложением "
                "в чат / ссылкой https.")
        r = requests.get(f"{turl}/files/read", params={"path": path},
                         headers={"Authorization": f"Bearer {tkey}"}, timeout=180)
        if r.status_code >= 400:
            raise MediaError(
                f"Файловое хранилище отдало {r.status_code} на чтение '{path}': {r.text[:200]}. "
                "Проверь, что путь абсолютный и файл существует (list_files/read_file хранилища).")
        ctype = (r.headers.get("content-type") or "").split(";")[0].strip()
        if ctype.startswith("image/") or _is_image_bytes(r.content):
            return r.content
        raise MediaError(
            f"Файл '{path}' прочитан, но это не картинка (content-type={ctype or 'text'}). "
            "Инструмент принимает только изображения.")

    def _resolve_media(self, value: str, files: list, request) -> str:
        """Превратить любой указатель (id файла чата / путь ФС / https / data URI / asset URL)
        в то, что принимает Krea: https URL, data URI или asset URL."""
        v = (value or "").strip()
        if not v:
            raise MediaError("Пустой указатель на изображение.")
        if v.startswith(("data:", "http://", "https://")):
            return v
        if _looks_like_file_id(v):
            data = self._fetch_chat_file_bytes(v, request, files)
            return _data_uri(data, hint_name=self._chat_filename(v, files))
        # иначе считаем путём в файловом хранилище
        try:
            data = self._fetch_terminal_file_bytes(v)
            return _data_uri(data, hint_name=v)
        except MediaError as e:
            # строка с '/' — точно путь, а не id файла чата: fallback не применяем
            if _looks_like_file_id(v) or "/" in v or "\\" in v:
                raise e
            # иначе попробуем как id чата (вдруг id не uuid-формата)
            try:
                data = self._fetch_chat_file_bytes(v, request, files)
                return _data_uri(data)
            except MediaError:
                raise e

    def _chat_filename(self, file_id: str, files: list) -> str:
        for f in files or []:
            if isinstance(f, dict) and (f.get("id") == file_id or f.get("url") == file_id):
                return f.get("name") or f.get("filename") or ""
        return ""

    def _first_chat_media(self, files: list, request) -> str:
        """Первое приложенное к сообщению изображение → data URI (через внутренний API OW)."""
        for f in files or []:
            if not isinstance(f, dict):
                continue
            if f.get("type", "file") != "file":
                continue
            fid = f.get("id") or f.get("url") or ""
            if not fid or str(fid).startswith(("http://", "https://", "data:")):
                # внешняя ссылка в вложении — можно отдать как есть
                if str(fid).startswith(("http://", "https://")):
                    return fid
                continue
            meta = f.get("meta") if isinstance(f.get("meta"), dict) else {}
            ctype = (f.get("content_type") or meta.get("content_type") or "")
            if ctype and not ctype.startswith("image/"):
                continue
            data = self._fetch_chat_file_bytes(fid, request, files)
            return _data_uri(data, hint_name=f.get("name") or f.get("filename") or "")
        return ""

    def _media_or_none(self, value: str, files: list, request) -> str:
        """Явный указатель побеждает; иначе первое вложение чата."""
        if (value or "").strip():
            return self._resolve_media(value, files, request)
        return self._first_chat_media(files, request)

    def _media_list(self, values: list, files: list, request) -> list:
        out = []
        for v in values or []:
            if isinstance(v, str) and v.strip():
                out.append(self._resolve_media(v, files, request))
        return out

    async def _run_job(self, route: str, payload: dict, wait_s: int, label: str, **_) -> str:
        """Submit POST /generate/{route}, poll until done, return URLs or job_id."""
        r = requests.post(f"{BASE_URL}generate/{route}", json=payload, headers=self._headers(), timeout=60)
        if r.status_code >= 400:
            return f"Krea submit error {r.status_code} ({route}): {r.text[:400]}"
        job_id = r.json()["job_id"]
        headers = self._headers()
        deadline = time.time() + max(10, wait_s)
        while True:
            pr = requests.get(f"{BASE_URL}jobs/{job_id}", headers=headers, timeout=60)
            if pr.status_code >= 400:
                return f"Krea poll error {pr.status_code}: {pr.text[:400]} (job_id={job_id})"
            job = pr.json()
            status = job.get("status")
            if status == "completed":
                urls = _extract_urls(job)
                return (f"{label} готово:\n" + "\n".join(urls)) if urls else f"Статус completed, но ссылок нет. job_id={job_id}"
            if status in ("failed", "cancelled"):
                err = (job.get("error") or {}).get("message") or status
                return f"{label} завершилась ошибкой: {err} (job_id={job_id})"
            if time.time() >= deadline:
                return f"{label} ещё выполняется. job_id={job_id} — проверь через get_job."
            await asyncio.sleep(POLL_INTERVAL_S)

    # ---------------------------------------------------- job / assets

    async def get_job(self, job_id: str) -> str:
        """
        Проверяет статус фонового задания Krea по job_id и возвращает результат:
        готовые ссылки result.urls или текущий статус (queued/processing/...).

        Вызывай этот метод, когда предыдущая генерация ответила
        «ещё выполняется ... job_id=...» — подставь этот же job_id сюда.

        :param job_id: UUID задачи, полученный при запуске генерации. Обязателен.
        """
        pr = requests.get(f"{BASE_URL}jobs/{job_id}", headers=self._headers(), timeout=60)
        if pr.status_code >= 400:
            return f"Krea error {pr.status_code}: {pr.text[:400]}"
        job = pr.json()
        if job.get("status") == "completed":
            urls = _extract_urls(job)
            return "Готово:\n" + "\n".join(urls) if urls else f"Статус completed, но ссылок нет. job_id={job_id}"
        return f"Статус: {job.get('status')}, job_id={job_id}"

    async def upload_asset(self, image: str = "", __files__: list = None, __request__: object = None) -> str:
        """
        Загружает изображение на сервер Krea (POST /assets) и возвращает asset URL
        (вида https://...), который затем можно подставлять в image/image_refs
        всех генерационных методов. Нужен, когда одну и ту же картинку предстоит
        использовать много раз (иначе каждый метод может принять указатель напрямую).

        [FILE_SOURCES]
        Если параметр image пуст — возьмётся первое изображение, приложенное к сообщению.

        :param image: указатель на изображение в любом виде: id файла чата (uuid из
            attached_files/list_chat_files), абсолютный путь файла в файловом хранилище
            (Open Terminal), https-URL или data URI. Примеры:
            image="9f2c1b...-uuid", image="/home/user/photo.png", image="https://.../a.jpg".
        :param __files__: файлы, приложенные пользователем к сообщению (инжектится Open WebUI
            автоматически — не заполняй сам).
        :param __request__: контекст запроса Open WebUI (инжектится автоматически — не заполняй).
        """
        try:
            media = self._media_or_none(image, __files__, __request__)
        except MediaError as e:
            return f"Error: {e}"
        if not media:
            return ("Error: не передано изображение и к сообщению ничего не приложено. "
                    "Передай id файла чата, путь из файлового хранилища или https-URL.")
        if media.startswith("data:"):
            raw = base64.b64decode(media.split(",", 1)[1])
            mime = _mime_by_magic(raw)
            name = "image." + ("jpg" if mime == "image/jpeg" else mime.split("/")[-1])
            files_kw = {"files": {"file": (name, io.BytesIO(raw), mime)}}
        else:
            # https/asset URL: скачаем и перезальём, чтобы получить стабильный asset URL
            rr = requests.get(media, timeout=180)
            if rr.status_code >= 400 or not _is_image_bytes(rr.content):
                return f"Error: не удалось скачать изображение по URL {media[:120]} (http {rr.status_code})."
            files_kw = {"files": {"file": ("image.png", io.BytesIO(rr.content), _mime_by_magic(rr.content))}}
        r = requests.post(f"{BASE_URL}assets", headers=self._headers(), timeout=180, **files_kw)
        if r.status_code >= 400:
            return f"Upload failed ({r.status_code}): {r.text[:400]}"
        a = r.json()
        return (f"Загружено. Asset URL: {a.get('image_url')} (id={a.get('id')}, "
                f"{a.get('width')}x{a.get('height')}). Дальше передавай этот URL в image/image_refs.")

    async def inspect_image_sources(self, __files__: list = None, __request__: object = None) -> str:
        """
        ДИАГНОСТИКА: показывает, какие изображения доступны прямо сейчас и как их передать:
        список вложений текущего чата (id, имя, content_type) и статус моста к файловому
        хранилищу (Open Terminal). Вызывай, если непонятно, какой указатель передать в
        image/image_refs, или если предыдущая передача файла завершилась ошибкой.

        [FILE_SOURCES]

        :param __files__: файлы, приложенные пользователем к сообщению (инжектится автоматически).
        :param __request__: контекст запроса Open WebUI (инжектится автоматически).
        """
        out = {"chat_files": [], "terminal_bridge": None, "how_to_pass": {}}
        for f in __files__ or []:
            if isinstance(f, dict):
                meta = f.get("meta") if isinstance(f.get("meta"), dict) else {}
                out["chat_files"].append({
                    "id": f.get("id") or f.get("url"),
                    "name": f.get("name") or f.get("filename"),
                    "content_type": f.get("content_type") or meta.get("content_type"),
                    "pass_as": f"image=\"{f.get('id') or f.get('url')}\"",
                })
        turl = (self.valves.terminal_url or "").strip().rstrip("/")
        if turl and self.valves.terminal_api_key:
            try:
                hr = requests.get(f"{turl}/health", headers={"Authorization": f"Bearer {self.valves.terminal_api_key}"}, timeout=10)
                out["terminal_bridge"] = f"{turl} → http {hr.status_code} (пути ФС принимаются в image/image_refs)"
            except Exception as e:
                out["terminal_bridge"] = f"{turl} → недоступен: {e}"
        else:
            out["terminal_bridge"] = "не настроен (Valves: terminal_url + terminal_api_key) — пути ФС не принимаются"
        out["how_to_pass"] = {
            "chat_attachment": "image=\"<id из chat_files выше>\"",
            "filesystem": "image=\"<абсолютный путь из list_files хранилища>\"",
            "web": "image=\"https://...\"",
        }
        return json.dumps(out, ensure_ascii=False, indent=2)

    # ---------------------------------------------------- TEXT → IMAGE

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
    ) -> str:
        """
        СОЗДАТЬ НОВОЕ ИЗОБРАЖЕНИЕ ИЗ ТЕКСТА (text→image).

        Вызывай, когда пользователь просит нарисовать/сгенерировать/сделать картинку с нуля
        («сгенерируй красивый закат», «нарисуй постер», «сделай логотип»), и редактировать
        готовый файл не требуется (для правок существующего фото — edit_image, для апскейла — enhance_image).
        Коллаж/композиция из нескольких фото пользователя: тоже сюда, с model='nano-banana-pro'
        (или 'gpt-image-2') и референсами в image_refs.

        [FILE_SOURCES]

        [IMAGE_MODEL_SELECTOR]

        :param prompt: подробное описание желаемой картинки на английском: сюжет, стиль,
            освещение, композиция. Например: "A serene ocean sunset, golden light, gentle waves, photorealistic".
            Перепиши формулировку пользователя, не отправляй её в сыром виде.
        :param model: модель Krea — смотри «Выбор модели» выше. По умолчанию 'seedream-4'.
        :param aspect_ratio: пропорции кадра: 1:1 (по умолчанию), 16:9, 9:16, 4:3, 3:2, 4:5, 2:3, 2:1, 1:2.
            «Горизонтальное/широкое» = 16:9, «вертикальное/портрет» = 9:16 или 4:5. Если пользователь
            просит точный размер в пикселях — используй exact_width/exact_height.
        :param resolution: масштаб выходной картинки: '1K' (по умолчанию), '2K' или '4K'.
            Поддерживают только nano-banana-* и gpt-image-*; на остальных моделях игнорируется.
        :param negative_prompt: что должно ОТСУТСТВОВАТЬ (watermark, text, blur). Поддерживается
            только qwen-2512; на других моделях игнорируется.
        :param image_refs: референсные изображения пользователя (до 10). Каждый элемент — указатель
            в любом виде: id файла чата, абсолютный путь файла в файловом хранилище (Open Terminal),
            https-URL, data URI или asset URL от upload_asset (см. раздел «КАК ПРАВИЛЬНО ПОЛУЧАТЬ
            ИЗОБРАЖЕНИЯ» выше). Передавай, когда пользователь просит «в стиле этого фото», «как на этой
            персоне», «коллаж из этих фото». Работает на nano-banana-*, gpt-image-*, krea-2-* (style ref),
            seedream-*, ideogram-3, grok-image, flux-kontext, z-image, muse, luma, runway.
        :param exact_width: точная ширина в px (256–4096). ONLY если пользователь назвал размеры; иначе 0.
        :param exact_height: точная высота в px (256–4096). ONLY если пользователь назвал размеры; иначе 0.
        :param seed: фиксированный seed (для повторения результата). 0 = случайный.
        :param wait_s: сколько секунд ждать результат (по умолчанию 180, максимум ~300).
            Если не дождались — вернётся job_id, проверь позже методом get_job.
        :param __files__: файлы из чата (инжектится Open WebUI автоматически — не заполняй сам).
            В generate_image вложения НЕ подставляются автоматически: референсы передавай явно
            через image_refs (id файла чата или путь из файлового хранилища).
        :param __request__: контекст запроса Open WebUI (инжектится автоматически — не заполняй).
        """
        try:
            route = _route(model)
        except ValueError as e:
            return f"Error: {e}"
        try:
            # референсы — только явные указатели в image_refs (id чата / путь ФС / URL);
            # молча подставлять вложение чата в text→image нельзя: это меняет смысл запроса
            media = self._media_list(image_refs, __files__, __request__)
        except MediaError as e:
            return f"Error: {e}"
        payload: dict = {"prompt": prompt}
        fields = MODEL_FIELDS.get(route, {"aspect_ratio", "resolution", "width", "height"})
        _size_or_ar(route, aspect_ratio, exact_width, exact_height, fields, payload)
        if resolution != "1K" and "resolution" in fields and not route.startswith(("image/krea/", "image/z-image/")):
            payload["resolution"] = resolution
        if negative_prompt and "negative_prompt" in fields:
            payload["negative_prompt"] = negative_prompt
        if seed and "seed" in fields:
            payload["seed"] = seed
        if media:
            if "image_urls" in fields:
                payload["image_urls"] = media[:10]
            else:
                placed = False
                for fname, routes in REF_FIELD_FALLBACK.items():
                    if route in routes and fname in fields:
                        if fname == "reference_images":  # runway: объекты {url, tag}
                            payload[fname] = [{"url": m, "tag": f"ref{i+1}"} for i, m in enumerate(media[:3])]
                        elif fname in ("style_images", "image_style_references"):
                            payload[fname] = [{"url": m, "strength": 0.8} for m in media[:10]]
                        else:
                            payload[fname] = media[:10]
                        placed = True
                        break
                if not placed:
                    return (f"Error: модель {_known(route)} не принимает референсные изображения. "
                            "Возьми nano-banana-pro, gpt-image-2, seedream-4, ideogram-3 или krea-2-large.")
        if route == "image/runway/gen-4-image" and not media:
            return ("Error: runway-gen-4 требует минимум один референс (reference_images). "
                    "Передай image_refs или возьми другую модель.")
        return await self._run_job(route, payload, wait_s, "Генерация")

    # ---------------------------------------------------- EDIT

    async def edit_image(
        self,
        prompt: str,
        image: str = "",
        additional_images: list = None,
        model: str = "seededit",
        strength: float = 0.0,
        wait_s: int = 180,
        __files__: list = None,
        __request__: object = None,
    ) -> str:
        """
        РЕДАКТИРОВАТЬ ГОТОВУЮ КАРТИНКУ ПО ТЕКСТОВОЙ ИНСТРУКЦИИ (image+text→image).

        Вызывай, когда пользователь ссылается на существующий файл/фото и просит ИЗМЕНИТЬ его
        («замени кошку на ворону», «убери фон», «сделай белый фон карточки товара», «нось костюм»,
        «отретушируй», «перенеси персонажа на закат»). Во всех этих случаях именно этот метод —
        правильный; генерировать «с нуля» не нужно даже если правки выглядят большими.
        Отдельный вызов upload_asset НЕ нужен: передай указатель на картинку напрямую в image.

        [FILE_SOURCES]

        :param prompt: ИНСТРУКЦИЯ по-английски, что изменить. Формула: глагол («replace/put/set/…»)
            + объект + сохраняемый контекст. Пример:
            "Replace the cat with a black raven sitting in the same spot, photorealistic,
            keep the background and lighting unchanged".
            (Не описание «как должно выглядеть в целом», а что именно ИЗМЕНИТЬ.)
        :param image: исходная картинка — указатель в любом виде: id файла чата (uuid из
            attached_files/list_chat_files), абсолютный путь файла в файловом хранилище
            (Open Terminal), https-URL, data URI или asset URL от upload_asset.
            Примеры: image="9f2c1b...-uuid", image="/home/user/photo.png".
            Если пуст — возьмётся первое изображение, приложенное к сообщению.
        :param additional_images: вспомогательные референсы (1–2) в том же формате указателей:
            например, фото вороны, чтобы именно её вставить. ПОДДЕРЖИВАЮТ только nano-banana-pro,
            gpt-image-2, grok-edit и flux-kontext (style); на других моделях будет ошибка 400 —
            выбирай model= соответственно.
        :param model: модель редактирования: 'seededit' (дефолт — замена объектов, ретушь),
            'flux-kontext' (сильная контекстная правка), 'nano-banana-pro' (умные правки,
            референсы и комбинирование нескольких фото), 'gpt-image-2' (гибкая инструкция),
            'grok-edit'.
        :param strength: сила изменения 0.0 − 1.0; актуален только для flux-kontext
            (0.4 — мягкая правка, 1.0 — полностью подчиниться инструкции). Для остальных 0.
        :param wait_s: секунды ожидания результата (как в generate_image).
        :param __files__: файлы из чата (инжектится Open WebUI). Если image пуст — метод возьмёт
            первое приложенное изображение сам.
        :param __request__: контекст запроса Open WebUI (инжектится автоматически — не заполняй).
        """
        try:
            media = self._media_or_none(image, __files__, __request__)
        except MediaError as e:
            return f"Error: {e}"
        if not media:
            return ("Error: не передана исходная картинка. Передай в image id файла чата "
                    "(uuid из attached_files/list_chat_files), путь файла из файлового хранилища, "
                    "https-URL или asset URL — либо приложи картинку к сообщению.")
        try:
            route = _route(model)
        except ValueError as e:
            return f"Error: {e}"
        try:
            extra = self._media_list(additional_images, __files__, __request__)
        except MediaError as e:
            return f"Error: {e}"
        fields = MODEL_FIELDS.get(route, {"image_url"})
        payload: dict = {"prompt": prompt}
        if "image_urls" in fields:
            payload["image_urls"] = ([media] + extra)[:10]
        elif "image_url" in fields:
            payload["image_url"] = media
        else:
            return f"Error: модель {_known(route)} не принимает исходную картинку. Возьми seededit, flux-kontext, nano-banana-pro, gpt-image-2 или grok-edit."
        if extra:
            if "image_urls" in fields:
                pass  # уже объединены выше
            elif "style_images" in fields:
                payload["style_images"] = [{"url": u, "strength": 0.7} for u in extra]
            elif "image_style_references" in fields:
                payload["image_style_references"] = [{"url": u, "strength": 0.7} for u in extra]
            else:
                return (f"Error: модель {_known(route)} не принимает доп. референсы. Используй "
                        "nano-banana-pro, gpt-image-2, grok-edit или flux-kontext.")
        if strength and "strength" in fields:
            payload["strength"] = max(0.0, min(1.0, strength))
        if route.startswith(("image/krea/", "image/z-image/")):
            # у krea-2/z-image aspect_ratio и resolution обязательны (enum resolution: 1K)
            allowed = (["1:1", "4:3", "3:2", "16:9", "2.35:1", "4:5", "2:3", "9:16"]
                       if route.startswith("image/krea/") else ["1:1", "4:3", "2:3", "16:9", "9:16"])
            src = _media_size(media)
            payload["aspect_ratio"] = _closest_aspect(src[0], src[1], allowed) if src else "1:1"
            payload["resolution"] = "1K"
        return await self._run_job(route, payload, wait_s, "Редактирование")

    # ---------------------------------------------------- ENHANCE / UPSCALE

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
        __request__: object = None,
    ) -> str:
        """
        АПСКЕЙЛ / ПОВЫШЕНИЕ КАЧЕСТВА И ДЕТАЛЬНОСТИ СУЩЕСТВУЮЩЕЙ КАРТИНКИ (enhance/upscale).

        Вызывай, когда пользователь просит «увеличь фото», «сделай 4K», «повысь разрешение»,
        «добавь резкости и деталей», «восстанови качество» — то есть СОХРАНИТЬ содержимое
        картинки, подняв её разрешение/детализацию. НЕ вызывай для рисования/правок — это
        generate_image / edit_image.

        Совместная задача «расширь влево на 30% И апскейл»: сначала edit_image — расширить
        кадр (запрос на правку), затем этот метод — апскейл.

        [FILE_SOURCES]

        :param image: исходная картинка — указатель в любом виде: id файла чата (uuid из
            attached_files/list_chat_files), абсолютный путь файла в файловом хранилище
            (Open Terminal), https-URL, data URI или asset URL от upload_asset.
            Если пуст — возьмётся первое изображение, приложенное к сообщению.
        :param target_width: ЖЕЛАЕМАЯ итоговая ширина в px (например 3840 для 4K). Если передана
            вместе с target_height, метод сам посчитает коэффициент масштабирования.
        :param target_height: желаемая итоговая высота в px. Обычно передаётся парой с target_width.
        :param scale_factor: во сколько раз увеличить (2 = вдвое). Используется, когда целевые
            width/height не заданы. krea-enhance: свободное значение ≥1; topaz-generative: 1–32.
        :param prompt: необязательная подсказка («sharper fur, cloud details») — творческая
            детализация конкретных элементов. Пусто = простая детализация без изменений сюжета.
        :param model: 'krea-enhance' (дёшево, креативно, до 8K) или 'topaz-generative'
            (максимум качества и контроля, до 16K).
        :param ai_strength: 0.1–1.0 — насколько смело дорисовывать новые детали (0.4 — умеренно).
            Применяется на krea-enhance.
        :param wait_s: секунды ожидания (дефолт 240 — апскейл медленный).
        :param __files__: файлы из чата (инжектится Open WebUI). Если image пуст — метод возьмёт
            первое приложенное изображение сам.
        :param __request__: контекст запроса Open WebUI (инжектится автоматически — не заполняй).
        """
        try:
            media = self._media_or_none(image, __files__, __request__)
        except MediaError as e:
            return f"Error: {e}"
        if not media:
            return ("Error: не передана картинка. Передай в image id файла чата, путь файла из "
                    "файлового хранилища, https-URL или asset URL — либо приложи картинку к сообщению.")
        try:
            route = _route(model)
        except ValueError as e:
            return f"Error: {e}"
        fields = MODEL_FIELDS.get(route, {"image_url", "prompt", "image_scaling_factor"})
        payload: dict = {"image_url": media}
        if "prompt" in fields:
            payload["prompt"] = prompt or ""
        factor = max(1.0, float(scale_factor or 2.0))
        if target_width and target_height:
            src = _media_size(media)
            if src:
                factor = max(target_width / src[0], target_height / src[1])
        if "topaz" in route:
            if target_width and target_height:
                payload["width"], payload["height"] = target_width, target_height
            else:
                src = _media_size(media)
                if not src:
                    return ("Error: topaz-generative требует целевые width/height, а размер исходной "
                            "картинки определить не удалось. Передай target_width/target_height.")
                payload["width"] = int(round(src[0] * factor))
                payload["height"] = int(round(src[1] * factor))
            payload["upscaling_activated"] = factor > 1.01
            payload["image_scaling_factor"] = min(32, max(1, round(factor, 2)))
        else:  # krea-enhance: только множитель
            payload["image_scaling_factor"] = round(max(1.0, factor), 2)
            if "ai_strength" in fields:
                payload["ai_strength"] = max(0.1, min(1.0, ai_strength or 0.4))
        return await self._run_job(route, payload, wait_s, "Апскейл")
