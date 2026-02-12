"""Chat service for orchestrating conversations and context building."""

import json
import re
import time
from datetime import datetime
from typing import Generator, Dict, Any, List, Optional

from app.models.base import BaseLLM
from app.services.memory_service import MemoryService
from app.data.database import db, Conversation, Message, UserProfile
from app.prompts import (
    get_mode_prefix,
    format_rolling_summary_prompt,
    format_title_summary_prompt
)
from utils.logger import get_logger
import config

logger = get_logger()


class ChatService:
    """
    Orchestrates chat operations including context building,
    response generation, and conversation management.
    """

    def __init__(
        self,
        llm: BaseLLM,
        memory: MemoryService,
        prompts: Dict[str, Any],
        gatekeeper=None
    ):
        """
        Initialize chat service.

        Args:
            llm: LLM provider for generation
            memory: Memory service for semantic memory
            prompts: Loaded prompt templates
            gatekeeper: Optional MemoryGatekeeper for classifying memory needs
        """
        self.llm = llm
        self.memory = memory
        self.prompts = prompts
        self.gatekeeper = gatekeeper

    def get_user_profile_context(self) -> str:
        """Build user profile prefix for AI context."""
        profile = UserProfile.query.first()
        if not profile or (not profile.name and not profile.details):
            return ''

        parts = ['[User Profile]']
        if profile.name:
            parts.append(f'Name: {profile.name}.')
        if profile.details:
            parts.append(f'About the user: {profile.details}')
        return ' '.join(parts) + '\n\n'

    def build_context(
        self,
        conversation_id: Optional[int],
        new_message: str,
        mode: str = 'normal',
        history: List[Dict[str, str]] = None
    ) -> str:
        """
        Build context using tiered memory system.

        Tiers:
        1. Tonality prefix (if mode != normal)
        2. User Profile
        3. Long-term memory (semantic retrieval)
        4. Rolling conversation summary
        5. Last 8 raw messages
        6. Current user message

        Args:
            conversation_id: ID of saved conversation (or None)
            new_message: Current user message
            mode: Response mode (normal, concise, logic)
            history: In-memory history for unsaved chats

        Returns:
            Assembled context string
        """
        context_parts = []

        # 1. Tonality prefix
        mode_prefix = get_mode_prefix(self.prompts, mode)
        if mode_prefix:
            context_parts.append(mode_prefix)

        # 2. User Profile
        user_context = self.get_user_profile_context()
        if user_context:
            context_parts.append(f"[User Profile]: {user_context}")

        # Build recent messages list early (needed for gatekeeper + context)
        recent_msgs_dicts = []
        if conversation_id:
            convo = Conversation.query.get(conversation_id)
            recent_db_msgs = Message.query.filter_by(conversation_id=conversation_id)\
                .order_by(Message.timestamp.desc()).limit(config.MAX_RECENT_MESSAGES).all()
            recent_db_msgs.reverse()
            recent_msgs_dicts = [
                {"role": msg.role, "content": msg.content} for msg in recent_db_msgs
            ]
        elif history:
            recent_msgs_dicts = history[-config.MAX_RECENT_MESSAGES:] if len(history) > config.MAX_RECENT_MESSAGES else history

        # 3. Long-term memory (gatekeeper decides if retrieval is needed)
        ltm = None
        if self.gatekeeper and config.GATEKEEPER_ENABLED:
            gatekeeper_context = recent_msgs_dicts[-3:] if recent_msgs_dicts else []
            classification = self.gatekeeper.classify(new_message, gatekeeper_context)

            if classification["memory_need"] in ("SEMANTIC", "MULTI"):
                query = " ".join(classification.get("retrieval_keys", [])) or new_message
                ltm = self.memory.get_relevant_facts(
                    query_text=query,
                    n_results=config.SEMANTIC_RESULTS_COUNT
                )
            # NONE, RECENT, PROFILE: skip semantic retrieval
        else:
            # No gatekeeper — always retrieve (original behavior)
            ltm = self.memory.get_relevant_facts(
                query_text=new_message,
                n_results=config.SEMANTIC_RESULTS_COUNT
            )

        if ltm:
            context_parts.append(f"[Relevant Memory]:\n{ltm}")

        # 4 & 5: Rolling summary and recent messages
        if conversation_id:
            if convo and convo.rolling_summary:
                context_parts.append(f"[Conversation Summary]: {convo.rolling_summary}")

            for msg_dict in recent_msgs_dicts:
                prefix = "User" if msg_dict['role'] == 'user' else "Assistant"
                context_parts.append(f"{prefix}: {msg_dict['content']}")
        elif history:
            for msg in recent_msgs_dicts:
                prefix = "User" if msg['role'] == 'user' else "Assistant"
                context_parts.append(f"{prefix}: {msg['content']}")

        # 6. Current message
        context_parts.append(f"User: {new_message}")
        context_parts.append("Assistant:")

        return "\n".join(context_parts)

    def generate_response(
        self,
        message: str,
        conversation_id: Optional[int] = None,
        mode: str = 'normal',
        history: List[Dict[str, str]] = None
    ) -> Generator[str, None, None]:
        """
        Generate streaming response for a chat message.

        Args:
            message: User message
            conversation_id: Conversation ID (or None for unsaved)
            mode: Response mode
            history: In-memory history for unsaved chats

        Yields:
            SSE-formatted data strings with tokens or completion signal
        """
        # Build context
        context = self.build_context(conversation_id, message, mode, history)

        try:
            # Stream tokens from LLM
            token_generator = self.llm.generate(context, stream=True)

            for token in token_generator:
                yield f"data: {json.dumps({'token': token})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    def auto_summarize(self, conversation_id: int) -> bool:
        """
        Auto-update rolling summary when enough new messages exist.

        Args:
            conversation_id: Conversation to summarize

        Returns:
            True if summarization occurred
        """
        convo = Conversation.query.get(conversation_id)
        if not convo:
            return False

        total_messages = Message.query.filter_by(conversation_id=conversation_id).count()
        messages_in_summary = convo.messages_summarized or 0
        unsummarized_count = total_messages - messages_in_summary

        if unsummarized_count < config.UNSUMMARIZED_THRESHOLD:
            return False

        # Get messages to summarize (excluding last 8 for raw context)
        all_msgs = Message.query.filter_by(conversation_id=conversation_id)\
            .order_by(Message.timestamp.asc()).all()

        messages_to_summarize = all_msgs[messages_in_summary:total_messages - config.MAX_RECENT_MESSAGES]

        if len(messages_to_summarize) < config.MIN_MESSAGES_TO_SUMMARIZE:
            return False

        new_messages_text = "\n".join([
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
            for m in messages_to_summarize
        ])

        existing_summary = convo.rolling_summary or "No previous summary."

        prompt = format_rolling_summary_prompt(
            self.prompts,
            existing_summary,
            new_messages_text
        )

        try:
            new_summary = self.llm.generate(prompt, stream=False)

            if new_summary:
                convo.rolling_summary = new_summary.strip()
                convo.messages_summarized = total_messages - config.MAX_RECENT_MESSAGES
                convo.last_summary_at = datetime.utcnow()
                db.session.commit()

                # Trigger semantic fact extraction from summarized messages
                try:
                    message_ids = [m.id for m in messages_to_summarize]
                    facts_stored = self.memory.process_semantic_memory(
                        conversation_id,
                        new_messages_text,
                        message_ids
                    )
                    if facts_stored > 0:
                        print(f"Stored {facts_stored} semantic facts from conversation {conversation_id}")
                except Exception as e:
                    print(f"Semantic memory extraction error: {e}")

                return True
        except Exception as e:
            print(f"Auto-summarization error: {e}")

        return False

    def save_chat_with_metadata(
        self,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Save conversation with AI-generated title and summary.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Dictionary with id, title, summary on success
            Dictionary with error on failure
        """
        if not messages:
            return {'error': 'No messages to save'}

        # Build prompt for title/summary generation
        history = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt = format_title_summary_prompt(self.prompts, history)

        try:
            response_text = self.llm.generate_json(prompt)

            # Try to parse JSON from response
            title = 'Untitled Chat'
            summary = ''

            if response_text:
                try:
                    json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group())
                        title = parsed.get('title', 'Untitled Chat')[:100]
                        summary = parsed.get('summary', '')
                except json.JSONDecodeError:
                    pass

            # Create conversation record
            convo = Conversation(title=title, summary=summary)
            db.session.add(convo)
            db.session.flush()  # Get the ID

            # Save all messages
            for msg in messages:
                m = Message(
                    conversation_id=convo.id,
                    role=msg['role'],
                    content=msg['content']
                )
                db.session.add(m)

            db.session.commit()

            return {
                'id': convo.id,
                'title': title,
                'summary': summary
            }

        except Exception as e:
            print(f"Error saving chat: {e}")
            return {'error': str(e)}
