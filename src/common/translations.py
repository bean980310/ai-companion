# translations.py

import locale
from typing import Dict, Optional, Union, List
import json
import os
from pathlib import Path
import logging
from presets import AI_ASSISTANT_PRESET, MINAMI_ASUKA_PRESET, MAKOTONO_AOI_PRESET, AINO_KOITO_PRESET

from src.common.default_language import default_language

logger = logging.getLogger(__name__)

class TranslationError(Exception):
    """번역 관련 커스텀 예외"""
    pass

class TranslationManager:
    def __init__(self, default_language: str = 'ko'):
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, str]] = {}
        self.character_settings: Dict[str, Dict[str, str]] = {
            'ai_assistant': AI_ASSISTANT_PRESET,
            'minami_asuka': MINAMI_ASUKA_PRESET,
            'makotono_aoi': MAKOTONO_AOI_PRESET,
            'aino_koito': AINO_KOITO_PRESET
        }
        self.load_translations()

    def load_translations(self) -> None:
        """번역 파일을 로드합니다."""
        try:
            translations_dir = Path('translations')
            translations_dir.mkdir(parents=True, exist_ok=True)
            
            if not list(translations_dir.glob('*.json')):
                self._create_default_translations()
            
            for lang_file in translations_dir.glob('*.json'):
                lang_code = lang_file.stem
                try:
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        self.translations[lang_code] = json.load(f)
                    logger.info(f"Loaded translations for {lang_code}")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error in {lang_file}: {e}")
                except Exception as e:
                    logger.error(f"Error loading translations for {lang_code}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to load translations: {e}")
            raise TranslationError(f"Failed to load translations: {e}")
                
    def get_character_setting(self, character: str = 'minami_asuka') -> str:
        """현재 언어의 캐릭터 설정 가져오기"""
        try:
            if character not in self.character_settings:
                logger.warning(f"Character {character} not found, using minami_asuka")
                character = 'minami_asuka'
            
            preset = self.character_settings[character]
            if self.current_language not in preset:
                logger.warning(
                    f"Language {self.current_language} not found for character {character}, "
                    f"using {self.default_language}"
                )
                return preset[self.default_language]
            
            return preset[self.current_language]
            
        except KeyError as e:
            error_msg = f"Character setting not found: {e}"
            logger.error(error_msg)
            raise TranslationError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error getting character setting: {e}"
            logger.error(error_msg)
            raise TranslationError(error_msg)
        
    def _create_default_translations(self):
        """기본 번역 파일 생성"""
        default_translations = {
            'ko': {
                'main_title': '간단한 Chatbot',
                'select_session_info': "선택된 세션이 표시됩니다.",
                'language_select': '언어 선택',
                'language_info': '인터페이스 언어를 선택하세요',
                'system_message': '시스템 메시지',
                'system_message_default': '당신은 유용한 AI 비서입니다.',
                'system_message_placeholder': '대화의 성격, 말투 등을 정의하세요.',
                'tab_main': '메인',
                'model_type_label': '모델 유형 선택',
                'model_select_label': '모델 선택',
                'character_select_label': "캐릭터 선택",
                'character_select_info': "대화할 캐릭터를 선택하세요.",
                'api_key_label': 'OpenAI/Anthropic API 키',
                'image_upload_label': '이미지 업로드 (선택)',
                'message_input_label': '메시지 입력',
                'message_placeholder': '메시지를 입력하세요...',
                'send_button': '전송',
                'advanced_setting': '고급 설정',
                'seed_label': '시드 값',
                'seed_info': '모델의 예측을 재현 가능하게 하기 위해 시드를 설정하세요.',
                "reset_session_button": "세션 초기화",
                "reset_all_sessions_button": "모든 세션 초기화",
                'download_tab': '다운로드',
                'download_title': '모델 다운로드',
                'download_description': 'HuggingFace에서 모델을 다운로드하고 로컬에 저장합니다.',
                'download_description_detail': '미리 정의된 모델 목록에서 선택하거나, 커스텀 모델 ID를 직접 입력할 수 있습니다.',
                'download_mode_label': '다운로드 방식 선택',
                'download_mode_predefined': '미리 정의된 모델',
                'download_mode_custom': '커스텀 모델 ID',
                'model_select_label': '모델 선택',
                'model_select_info': '지원되는 모델 목록입니다.',
                'custom_model_id_label': '커스텀 모델 ID',
                'custom_model_id_placeholder': '예) facebook/opt-350m',
                'custom_model_id_info': 'HuggingFace의 모델 ID를 입력하세요 (예: organization/model-name)',
                'save_path_label': '저장 경로',
                'save_path_placeholder': './models/my-model',
                'save_path_info': '비워두면 자동으로 경로가 생성됩니다.',
                'auth_required_label': '인증 필요',
                'auth_required_info': '비공개 또는 gated 모델 다운로드 시 체크',
                'hf_token_label': 'HuggingFace Token',
                'hf_token_placeholder': 'hf_...',
                'hf_token_info': 'HuggingFace에서 발급받은 토큰을 입력하세요.',
                'download_start_button': '다운로드 시작',
                'download_cancel_button': '취소',
                'download_details_label': '상세 정보',
                'download_log_label': '다운로드 로그',
                'download_preparing': '🔄 다운로드 준비 중...',
                'download_in_progress': '🔄 다운로드 중...',
                'download_complete': '✅ 다운로드 완료!',
                'download_failed': '❌ 다운로드 실패',
                'download_error': '❌ 오류 발생',
                'download_error_no_model': '❌ 모델 ID를 입력해주세요.',
                'hub_tab_title': '허브',
                'hub_description': '허깅페이스 허브 모델 검색',
                'hub_description_detail': '허깅페이스 허브에서 모델을 검색하고 다운로드할 수 있습니다.',
                'hub_search_description': '키워드로 검색하거나 필터를 사용하여 원하는 모델을 찾을 수 있습니다.',
                'hub_search_label': '검색어',
                'hub_search_placeholder': '모델 이름, 태그 또는 키워드를 입력하세요',
                'hub_search_button': '검색',
                'hub_model_type_label': '모델 유형',
                'hub_language_label': '언어',
                'hub_library_label': '라이브러리',
                'hub_model_list_label': '검색 결과',
                'hub_selected_model_label': '선택된 모델',
                'hub_save_path_label': '저장 경로',
                'hub_save_path_placeholder': './models/my-model',
                'hub_save_path_info': '비워두면 자동으로 경로가 생성됩니다.',
                'hub_auth_required_label': '인증 필요',
                'hub_auth_required_info': '비공개 또는 gated 모델 다운로드 시 체크',
                'hub_token_label': 'HuggingFace Token',
                'hub_token_placeholder': 'hf_...',
                'hub_token_info': 'HuggingFace에서 발급받은 토큰을 입력하세요.',
                'hub_download_button': '다운로드',
                'hub_cancel_button': '취소',
                'hub_details_label': '상세 정보',
                'hub_download_log_label': '다운로드 로그',
                "cache_tab_title": "캐시",
                "refresh_model_list_button": "모델 목록 새로고침",
                "refresh_info_label": "새로고침 결과",
                "cache_clear_all_button": "모든 모델 캐시 삭제",
                "clear_all_result_label": "결과",
                "handle_character_change_invalid_error": "❌ 선택한 캐릭터가 유효하지 않습니다."
                # 추가 번역키들...
            },
            'ja': {
                'main_title': 'シンプルなチャットボット',
                'select_session_info': "選択されたセッションが表示されます。",
                'language_select': '言語選択',
                'language_info': 'インターフェース言語を選択してください',
                'system_message': 'システムメッセージ',
                'system_message_default': 'あなたは有用なAI秘書です。',
                'system_message_placeholder': '会話の性格、話し方などを定義してください。',
                'tab_main': 'メイン',
                'model_type_label': 'モデルタイプの選択',
                'model_select_label': 'モデルの選択',
                'character_select_label': "キャラクター選択",
                'character_select_info': "会話するキャラクターを選択してください。",
                'api_key_label': 'OpenAI/Anthropic APIキー',
                'image_upload_label': '画像のアップロード（オプション）',
                'message_input_label': 'メッセージを入力',
                'message_placeholder': 'メッセージを入力してください...',
                'send_button': '送信',
                'advanced_setting': '詳細設定',
                'seed_label': 'シード値',
                'seed_info': 'モデルの予測を再現可能にするためにシード値を設定してください。',
                "reset_session_button": "セッション初期化",
                "reset_all_sessions_button": "すべてのセッションの初期化",
                'download_tab': 'ダウンロード',
                'download_title': 'モデルダウンロード',
                'download_description': 'HuggingFaceからモデルをダウンロードしてローカルに保存します。',
                'download_description_detail': '事前定義されたモデルリストから選択するか、カスタムモデルIDを直接入力できます。',
                'download_mode_label': 'ダウンロード方式',
                'download_mode_predefined': '事前定義モデル',
                'download_mode_custom': 'カスタムモデルID',
                'model_select_label': 'モデル選択',
                'model_select_info': 'サポートされているモデルのリストです。',
                'custom_model_id_label': 'カスタムモデルID',
                'custom_model_id_placeholder': '例）facebook/opt-350m',
                'custom_model_id_info': 'HuggingFaceのモデルIDを入力してください（例：organization/model-name）',
                'save_path_label': '保存パス',
                'save_path_placeholder': './models/my-model',
                'save_path_info': '空欄の場合、自動的にパスが生成されます。',
                'auth_required_label': '認証必要',
                'auth_required_info': 'プライベートまたはゲート付きモデルのダウンロード時にチェック',
                'hf_token_label': 'HuggingFaceトークン',
                'hf_token_placeholder': 'hf_...',
                'hf_token_info': 'HuggingFaceで発行されたトークンを入力してください。',
                'download_start_button': 'ダウンロード開始',
                'download_cancel_button': 'キャンセル',
                'download_details_label': '詳細情報',
                'download_log_label': 'ダウンロードログ',
                'download_preparing': '🔄 ダウンロード準備中...',
                'download_in_progress': '🔄 ダウンロード中...',
                'download_complete': '✅ ダウンロード完了！',
                'download_failed': '❌ ダウンロード失敗',
                'download_error': '❌ エラーが発生しました',
                'download_error_no_model': '❌ モデルIDを入力してください。',
                'hub_tab_title': 'ハブ',
                'hub_description': 'HuggingFaceハブモデル検索',
                'hub_description_detail': 'HuggingFaceハブからモデルを検索してダウンロードできます。',
                'hub_search_description': 'キーワード検索やフィルターを使用して目的のモデルを見つけることができます。',
                'hub_search_label': '検索ワード',
                'hub_search_placeholder': 'モデル名、タグ、またはキーワードを入力してください',
                'hub_search_button': '検索',
                'hub_model_type_label': 'モデルタイプ',
                'hub_language_label': '言語',
                'hub_library_label': 'ライブラリ',
                'hub_model_list_label': '検索結果',
                'hub_selected_model_label': '選択されたモデル',
                'hub_save_path_label': '保存パス',
                'hub_save_path_placeholder': './models/my-model',
                'hub_save_path_info': '空欄の場合、自動的にパスが生成されます。',
                'hub_auth_required_label': '認証必要',
                'hub_auth_required_info': 'プライベートまたはゲート付きモデルのダウンロード時にチェック',
                'hub_token_label': 'HuggingFaceトークン',
                'hub_token_placeholder': 'hf_...',
                'hub_token_info': 'HuggingFaceで発行されたトークンを入力してください。',
                'hub_download_button': 'ダウンロード',
                'hub_cancel_button': 'キャンセル',
                'hub_details_label': '詳細情報',
                'hub_download_log_label': 'ダウンロードログ',
                "cache_tab_title": "キャッシュ",
                "refresh_model_list_button": "モデルリスト更新",
                "refresh_info_label": "更新結果",
                "cache_clear_all_button": "すべてのモデルキャッシュを削除",
                "clear_all_result_label": "結果",
                "handle_character_change_invalid_error": "❌ 選択したキャラクターが無効です。"
                # 追加の翻訳キー...
            },
            'zh_CN': {
                'main_title': '简单聊天机器人',
                'select_session_info': "此时将显示选定的会话。",
                'language_select': '选择语言',
                'language_info': '选择界面语言',
                'system_message': '系统消息',
                'system_message_default': '你是一个有用的AI秘书。',
                'system_message_placeholder': '定义对话特征、语气等。',
                'tab_main': '主干',
                'model_type_label': '选择模型类型',
                'model_select_label': '选择模型',
                'character_select_label': "角色选择",
                'character_select_info': "请选择要对话的角色。",
                'api_key_label': 'OpenAI/Anthropic API密钥',
                'image_upload_label': '上传图片（可选）',
                'message_input_label': '输入消息',
                'message_placeholder': '请输入消息...',
                'send_button': '发送',
                'advanced_setting': '高级设置',
                'seed_label': '种子值',
                'seed_info': '设置种子值以使模型预测可重现。',
                "reset_session_button": "会话初始化",
                "reset_all_sessions_button": "初始化所有会话",
                'download_tab': '下载',
                'download_title': '模型下载',
                'download_description': '从HuggingFace下载模型并保存到本地。',
                'download_description_detail': '您可以从预定义模型列表中选择，或直接输入自定义模型ID。',
                'download_mode_label': '下载方式',
                'download_mode_predefined': '预定义模型',
                'download_mode_custom': '自定义模型ID',
                'model_select_label': '选择型号',
                'model_select_info': '支持的模型列表。',
                'custom_model_id_label': '自定义模型ID',
                'custom_model_id_placeholder': '例如：facebook/opt-350m',
                'custom_model_id_info': '输入HuggingFace模型ID（例如：organization/model-name）',
                'save_path_label': '保存路径',
                'save_path_placeholder': './models/my-model',
                'save_path_info': '留空将自动生成路径。',
                'auth_required_label': '需要认证',
                'auth_required_info': '下载私有或受限模型时勾选',
                'hf_token_label': 'HuggingFace令牌',
                'hf_token_placeholder': 'hf_...',
                'hf_token_info': '输入您的HuggingFace令牌。',
                'download_start_button': '开始下载',
                'download_cancel_button': '取消',
                'download_details_label': '详细信息',
                'download_log_label': '下载日志',
                'download_preparing': '🔄 准备下载中...',
                'download_in_progress': '🔄 下载中...',
                'download_complete': '✅ 下载完成！',
                'download_failed': '❌ 下载失败',
                'download_error': '❌ 发生错误',
                'download_error_no_model': '❌ 请输入模型ID。',
                'hub_tab_title': '中心',
                'hub_description': 'HuggingFace中心模型搜索',
                'hub_description_detail': '从HuggingFace中心搜索和下载模型。',
                'hub_search_description': '可以使用关键词搜索或使用过滤器找到所需的模型。',
                'hub_search_label': '搜索词',
                'hub_search_placeholder': '输入模型名称、标签或关键词',
                'hub_search_button': '搜索',
                'hub_model_type_label': '模型类型',
                'hub_language_label': '语言',
                'hub_library_label': '库',
                'hub_model_list_label': '搜索结果',
                'hub_selected_model_label': '已选模型',
                'hub_save_path_label': '保存路径',
                'hub_save_path_placeholder': './models/my-model',
                'hub_save_path_info': '留空将自动生成路径。',
                'hub_auth_required_label': '需要认证',
                'hub_auth_required_info': '下载私有或受限模型时勾选',
                'hub_token_label': 'HuggingFace令牌',
                'hub_token_placeholder': 'hf_...',
                'hub_token_info': '输入您的HuggingFace令牌。',
                'hub_download_button': '下载',
                'hub_cancel_button': '取消',
                'hub_details_label': '详细信息',
                'hub_download_log_label': '下载日志',
                "cache_tab_title": "高速缓存",
                "refresh_model_list_button": "更新模型列表",
                "refresh_info_label": "刷新结果",
                "cache_clear_all_button": "删除所有模型缓存",
                "clear_all_result_label": "结果",
                "handle_character_change_invalid_error": "❌ 选中的角色无效。"
                # 其他翻译键...
            },
            'zh_TW': {
                'main_title': '簡單聊天機器人',
                'select_session_info': "此時將顯示選定的會話。",
                'language_select': '選擇語言',
                'language_info': '選擇界面語言',
                'system_message': '系統消息',
                'system_message_default': '你是一個有用的AI祕書。',
                'system_message_placeholder': '定義對話特徵、語氣等。',
                'tab_main': '主幹',
                'model_type_label': '選擇模型類型',
                'model_select_label': '選擇模型',
                'character_select_label': "角色選擇",
                'character_select_info': "請選擇要對話的角色。",
                'api_key_label': 'OpenAI/Anthropic API密鑰',
                'image_upload_label': '上傳圖片（可選）',
                'message_input_label': '輸入消息',
                'message_placeholder': '請輸入消息...',
                'send_button': '發送',
                'advanced_setting': '高級設置',
                'seed_label': '種子值',
                'seed_info': '設置種子值以使模型預測可重現。',
                "reset_session_button": "會話初始化",
                "reset_all_sessions_button": "初始化所有會話",
                'download_tab': '下載',
                'download_title': '模型下載',
                'download_description': '從HuggingFace下載模型並儲存到本地。',
                'download_description_detail': '您可以從預定義模型列表中選擇，或直接輸入自訂模型ID。',
                'download_mode_label': '下載方式',
                'download_mode_predefined': '預定義模型',
                'download_mode_custom': '自訂模型ID',
                'model_select_label': '選擇型號',
                'model_select_info': '支援的模型列表。',
                'custom_model_id_label': '自訂模型ID',
                'custom_model_id_placeholder': '例如：facebook/opt-350m',
                'custom_model_id_info': '輸入HuggingFace模型ID（例如：organization/model-name）',
                'save_path_label': '儲存路徑',
                'save_path_placeholder': './models/my-model',
                'save_path_info': '留空將自動生成路徑。',
                'auth_required_label': '需要認證',
                'auth_required_info': '下載私有或受限模型時勾選',
                'hf_token_label': 'HuggingFace權杖',
                'hf_token_placeholder': 'hf_...',
                'hf_token_info': '輸入您的HuggingFace權杖。',
                'download_start_button': '開始下載',
                'download_cancel_button': '取消',
                'download_details_label': '詳細信息',
                'download_log_label': '下載日誌',
                'download_preparing': '🔄 準備下載中...',
                'download_in_progress': '🔄 下載中...',
                'download_complete': '✅ 下載完成！',
                'download_failed': '❌ 下載失敗',
                'download_error': '❌ 發生錯誤',
                'download_error_no_model': '❌ 請輸入模型ID。',
                'hub_tab_title': '中心',
                'hub_description': 'HuggingFace中心模型搜尋',
                'hub_description_detail': '從HuggingFace中心搜尋和下載模型。',
                'hub_search_description': '可以使用關鍵詞搜尋或使用過濾器找到所需的模型。',
                'hub_search_label': '搜尋詞',
                'hub_search_placeholder': '輸入模型名稱、標籤或關鍵詞',
                'hub_search_button': '搜尋',
                'hub_model_type_label': '模型類型',
                'hub_language_label': '語言',
                'hub_library_label': '庫',
                'hub_model_list_label': '搜尋結果',
                'hub_selected_model_label': '已選模型',
                'hub_save_path_label': '儲存路徑',
                'hub_save_path_placeholder': './models/my-model',
                'hub_save_path_info': '留空將自動生成路徑。',
                'hub_auth_required_label': '需要認證',
                'hub_auth_required_info': '下載私有或受限模型時勾選',
                'hub_token_label': 'HuggingFace權杖',
                'hub_token_placeholder': 'hf_...',
                'hub_token_info': '輸入您的HuggingFace權杖。',
                'hub_download_button': '下載',
                'hub_cancel_button': '取消',
                'hub_details_label': '詳細資訊',
                'hub_download_log_label': '下載日誌',
                "cache_tab_title": "高速緩存",
                "refresh_model_list_button": "更新模型列表",
                "refresh_info_label": "刷新結果",
                "cache_clear_all_button": "刪除所有模型緩存",
                "clear_all_result_label": "結果",
                "handle_character_change_invalid_error": "❌ 選中的角色無效。"
            },
            'en': {
                'main_title': 'Simple Chatbot',
                'select_session_info': "The selected session is displayed.",
                'language_select': 'Select Language',
                'language_info': 'Choose interface language',
                'system_message': 'System Message',
                'system_message_default': 'You are a useful AI assistant.',
                'system_message_placeholder': 'Define conversation characteristics, tone, etc.',
                'tab_main': 'Main',
                'model_type_label': 'Select Model Type',
                'model_select_label': 'Select Model',
                'character_select_label': "Character selection",
                'character_select_info': "Select the character you want to talk to.",
                'api_key_label': 'OpenAI/Anthropic API Key',
                'image_upload_label': 'Upload Image (Optional)',
                'message_input_label': 'Enter Message',
                'message_placeholder': 'Type your message...',
                'send_button': 'Send',
                'advanced_setting': 'Advanced Settings',
                'seed_label': 'Seed Value',
                'seed_info': 'Set a seed value to make model predictions reproducible.',
                "reset_session_button": "Reset Session",
                "reset_all_sessions_button": "Reset all Sessions",
                'download_tab': 'Download',
                'download_title': 'Model Download',
                'download_description': 'Download models from HuggingFace and save them locally.',
                'download_description_detail': 'You can select from a predefined list of models or directly enter a custom model ID.',
                'download_mode_label': 'Download Method',
                'download_mode_predefined': 'Predefined Models',
                'download_mode_custom': 'Custom Model ID',
                'model_select_label': 'Model Select',
                'model_select_info': 'List of supported models.',
                'custom_model_id_label': 'Custom Model ID',
                'custom_model_id_placeholder': 'e.g., facebook/opt-350m',
                'custom_model_id_info': 'Enter a HuggingFace model ID (e.g., organization/model-name)',
                'save_path_label': 'Save Path',
                'save_path_placeholder': './models/my-model',
                'save_path_info': 'Leave empty for automatic path generation.',
                'auth_required_label': 'Authentication Required',
                'auth_required_info': 'Check for private or gated model downloads',
                'hf_token_label': 'HuggingFace Token',
                'hf_token_placeholder': 'hf_...',
                'hf_token_info': 'Enter your HuggingFace token.',
                'download_start_button': 'Start Download',
                'download_cancel_button': 'Cancel',
                'download_details_label': 'Details',
                'download_log_label': 'Download Log',
                'download_preparing': '🔄 Preparing download...',
                'download_in_progress': '🔄 Downloading...',
                'download_complete': '✅ Download complete!',
                'download_failed': '❌ Download failed',
                'download_error': '❌ Error occurred',
                'download_error_no_model': '❌ Please enter a model ID.',
                'hub_tab_title': 'Hub',
                'hub_description': 'HuggingFace Hub Model Search',
                'hub_description_detail': 'Search and download models from HuggingFace Hub.',
                'hub_search_description': 'You can find desired models using keywords or filters.',
                'hub_search_label': 'Search',
                'hub_search_placeholder': 'Enter model name, tags, or keywords',
                'hub_search_button': 'Search',
                'hub_model_type_label': 'Model Type',
                'hub_language_label': 'Language',
                'hub_library_label': 'Library',
                'hub_model_list_label': 'Search Results',
                'hub_selected_model_label': 'Selected Model',
                'hub_save_path_label': 'Save Path',
                'hub_save_path_placeholder': './models/my-model',
                'hub_save_path_info': 'Leave empty for automatic path generation.',
                'hub_auth_required_label': 'Authentication Required',
                'hub_auth_required_info': 'Check for private or gated model downloads',
                'hub_token_label': 'HuggingFace Token',
                'hub_token_placeholder': 'hf_...',
                'hub_token_info': 'Enter your HuggingFace token.',
                'hub_download_button': 'Download',
                'hub_cancel_button': 'Cancel',
                'hub_details_label': 'Details',
                'hub_download_log_label': 'Download Log',
                "cache_tab_title": "Cache",
                "refresh_model_list_button": "Refresh Model List",
                "refresh_info_label": "Refresh Results",
                "cache_clear_all_button": "Delete all model caches",
                "clear_all_result_label": "Results",
                "handle_character_change_invalid_error": "❌ The selected character is invalid."
                # Additional translation keys...
            }
        }
        
        default_presets = {
            'AI_ASSISTANT': AI_ASSISTANT_PRESET,
            'MINAMI_ASUKA': MINAMI_ASUKA_PRESET,
            'MAKOTONO_AOI': MAKOTONO_AOI_PRESET,
            'AINO_KOITO': AINO_KOITO_PRESET
        }

        for lang, translations in default_translations.items():
            file_path = Path('translations') / f'{lang}.json'
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)
            logger.info(f"Created default translations for {lang}")
            
        
        for preset_name, languages in default_presets.items():
            for lang_code, content in languages.items():
                self.character_settings[preset_name.lower()][lang_code] = content

    def set_language(self, language_code: str) -> bool:
        """현재 언어 설정"""
        if language_code in self.translations:
            self.current_language = language_code
            logger.info(f"Language changed to {language_code}")
            return True
        logger.warning(f"Language {language_code} not found, using default language")
        return False

    def get(self, key: str, **kwargs) -> str:
        """UI 텍스트 번역 가져오기"""
        try:
            translation = self.translations[self.current_language].get(
                key,
                self.translations[self.default_language].get(key, key)
            )
            return translation.format(**kwargs) if kwargs else translation
        except KeyError:
            logger.warning(f"Translation key not found: {key}")
            return key
        except Exception as e:
            logger.error(f"Translation error for key '{key}': {e}")
            return key

    def get_available_languages(self) -> list:
        """사용 가능한 언어 코드 목록"""
        return list(self.translations.keys())

    def get_language_code(self, display_name: str) -> str:
        """디스플레이 이름을 언어 코드로 변환"""
        language_map = {
            '한국어': 'ko',
            'English': 'en',
            '日本語': 'ja',
            '中文(简体)': 'zh',
            '中文(繁體)': 'zh-tw',
            # 필요 시 추가 언어 매핑
        }
        language_code = language_map.get(display_name)
        if not language_code:
            logger.warning(f"알 수 없는 언어 선택: {display_name}. 기본 언어 '{self.default_language}' 사용.")
            language_code = self.default_language
        return language_code
    
    def get_available_languages_display_names(self) -> List[str]:
        """사용 가능한 언어의 디스플레이 이름 목록을 반환"""
        return list({
            '한국어': 'ko',
            'English': 'en',
            '日本語': 'ja',
            '中文(简体)': 'zh',
            '中文(繁體)': 'zh-tw',
            # 필요 시 추가 언어
        }.keys())

    
    def get_language_display_name(self, lang_code: str) -> str:
        """언어 코드에 대한 표시 이름"""
        display_names = {
            'ko': '한국어',
            'ja': '日本語',
            'zh_CN': '中文(简体)',
            'zh_TW': '中文(繁體)',
            'en': 'English'
        }
        return display_names.get(lang_code, lang_code)


translation_manager = TranslationManager(default_language=default_language)

# 간편한 접근을 위한 헬퍼 함수
def _(key: str, **kwargs) -> str:
    """UI 텍스트 번역을 위한 단축 함수"""
    return translation_manager.get(key, **kwargs)

def get_character_message(character: str = 'minami_asuka') -> str:
    """캐릭터 설정 메시지 반환"""
    return translation_manager.get_character_setting(character)