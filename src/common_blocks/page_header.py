# page_header.py
# 멀티페이지 대응 공통 헤더 컴포넌트

import gradio as gr
from typing import Optional, Tuple, Callable, List
from dataclasses import dataclass, field

from ..common.translations import translation_manager, _
from ..common.default_language import default_language


# 언어 코드 매핑
LANGUAGE_CODE_MAP = {
    "한국어": "ko",
    "日本語": "ja",
    "中文(简体)": "zh_CN",
    "中文(繁體)": "zh_TW",
    "English": "en"
}

LANGUAGE_CHOICES = ["한국어", "日本語", "中文(简体)", "中文(繁體)", "English"]


@dataclass
class PageHeaderComponents:
    """페이지 헤더 컴포넌트들을 담는 데이터클래스"""
    header_row: gr.Row = None
    title: gr.Markdown = None
    subtitle: gr.Markdown = None
    language_dropdown: gr.Dropdown = None


def create_page_header(
    page_title_key: str = "main_title",
    show_subtitle: bool = True,
    subtitle_text: str = "### Beta Release"
) -> PageHeaderComponents:
    """
    멀티페이지용 공통 헤더를 생성합니다.

    Args:
        page_title_key: 번역 키 (translations/*.json의 키)
        show_subtitle: 부제목 표시 여부
        subtitle_text: 부제목 텍스트

    Returns:
        PageHeaderComponents: 생성된 헤더 컴포넌트들
    """
    with gr.Row(elem_classes="header-container") as header_row:
        with gr.Column(scale=3):
            title = gr.Markdown(
                f"## {_(page_title_key)}",
                elem_classes="title"
            )
            if show_subtitle:
                subtitle = gr.Markdown(subtitle_text)
            else:
                subtitle = None

        with gr.Column(scale=1, min_width=150):
            language_dropdown = gr.Dropdown(
                label=_('language_select'),
                choices=LANGUAGE_CHOICES,
                value=translation_manager.get_language_display_name(default_language),
                interactive=True,
                info=_('language_info'),
                container=False,
                elem_classes="language-selector"
            )

    return PageHeaderComponents(
        header_row=header_row,
        title=title,
        subtitle=subtitle,
        language_dropdown=language_dropdown
    )


def get_language_code(display_name: str) -> str:
    """디스플레이 이름을 언어 코드로 변환"""
    return LANGUAGE_CODE_MAP.get(display_name, default_language)


def create_language_change_handler(
    components_to_update: List[Tuple[gr.Component, str, Optional[dict]]]
) -> Callable:
    """
    언어 변경 핸들러를 생성합니다.

    Args:
        components_to_update: 업데이트할 컴포넌트 목록
            각 튜플은 (컴포넌트, 번역키, 추가옵션) 형태

    Returns:
        언어 변경 시 호출될 함수
    """
    def handle_language_change(selected_lang: str):
        lang_code = get_language_code(selected_lang)
        translation_manager.set_language(lang_code)

        updates = []
        for component, trans_key, extra_opts in components_to_update:
            update_dict = {"value": _(trans_key)}
            if extra_opts:
                update_dict.update(extra_opts)
            updates.append(gr.update(**update_dict))

        return updates

    return handle_language_change


def setup_language_change_event(
    language_dropdown: gr.Dropdown,
    title: gr.Markdown,
    page_title_key: str = "main_title",
    additional_outputs: List[gr.Component] = None,
    additional_update_fn: Callable = None
):
    """
    언어 변경 이벤트를 설정합니다.

    Args:
        language_dropdown: 언어 선택 드롭다운
        title: 타이틀 마크다운 컴포넌트
        page_title_key: 페이지 타이틀 번역 키
        additional_outputs: 추가로 업데이트할 컴포넌트 목록
        additional_update_fn: 추가 업데이트 함수 (언어 코드를 인자로 받음)
    """
    outputs = [title, language_dropdown]
    if additional_outputs:
        outputs.extend(additional_outputs)

    def on_language_change(selected_lang: str):
        lang_code = get_language_code(selected_lang)
        translation_manager.set_language(lang_code)

        # 기본 업데이트
        results = [
            gr.update(value=f"## {_(page_title_key)}"),
            gr.update(label=_('language_select'), info=_('language_info'))
        ]

        # 추가 업데이트가 있으면 실행
        if additional_update_fn and additional_outputs:
            extra_results = additional_update_fn(lang_code)
            if isinstance(extra_results, (list, tuple)):
                results.extend(extra_results)
            else:
                results.append(extra_results)

        return results

    language_dropdown.change(
        fn=on_language_change,
        inputs=[language_dropdown],
        outputs=outputs
    )
