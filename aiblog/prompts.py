from __future__ import annotations

from dataclasses import dataclass

from aiblog.config import StyleConfig


@dataclass(frozen=True)
class PromptPack:
    system: str
    outline: str
    draft: str
    rewrite: str
    headline: str


def build_prompts(style: StyleConfig) -> PromptPack:
    rules = "\n".join([f"- {r}" for r in style.voice_rules])
    system = f"""Ты — помощник редактора блога. Пиши в стиле автора, соблюдая правила:\n{rules}\n\nВажно:\n- Используй только факты из предоставленного контекста и заметок.\n- Если фактов не хватает, помечай как TODO (не выдумывай).\n"""

    outline = """Сделай структурированный план поста на тему.\n\nТема: {topic}\n\nЗаметки автора (если есть):\n{notes}\n\nКонтекст из прошлых постов:\n{context}\n\nФормат:\n- Заголовок (1 вариант)\n- Лид (2-4 предложения)\n- Секция 1/2/3 (тезисы)\n- Примеры/кейсы (если уместно)\n- Вывод\n"""

    draft = """Напиши черновик поста.\n\nТема: {topic}\n\nЗаметки автора (если есть):\n{notes}\n\nКонтекст из прошлых постов:\n{context}\n\nТребования:\n- Не копируй дословно большие куски контекста; используй его как основу.\n- Если не хватает данных для конкретики — TODO.\n- В конце добавь блок \"Следующие шаги\" (2-5 пунктов).\n"""

    rewrite = """Перепиши текст в стиле автора.\n\nТекст для переписывания:\n{input_text}\n\nОпорный контекст (если помогает сохранить факты/термины):\n{context}\n\nТребования:\n- Сохрани смысл и факты.\n- Улучши ясность, структуру, убери воду.\n"""

    headline = """Дай 10 вариантов заголовков под тему.\n\nТема: {topic}\n\nКонтекст:\n{context}\n\nТребования:\n- Вариативность (серьёзные/цепляющие/прагматичные)\n- Без кликбейта\n"""

    return PromptPack(system=system, outline=outline, draft=draft, rewrite=rewrite, headline=headline)

