"""Flask application entry point for Joey-Bot."""

from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from sqlalchemy import func

from app.data.database import db, Conversation, Message, UserProfile, TokenUsage, init_db
from app.models.ollama_wrapper import OllamaWrapper
from app.services.chat_service import ChatService
from app.services.memory_service import MemoryService
from app.prompts import load_prompts
from utils.logger import setup_logging, get_logger, log_token_usage as log_token
import config

# Setup logging
setup_logging()
logger = get_logger()

# Create Flask app
app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = config.DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
init_db(app)

# Load prompts from YAML
prompts = load_prompts()

# Initialize services (dependency injection)
llm = OllamaWrapper(
    model=config.LITELLM_CHAT_MODEL,
    embedding_model=config.LITELLM_EMBEDDING_MODEL,
    api_base="http://localhost:11434"
)
memory_service = MemoryService(llm, prompts, config.VECTOR_STORE_PATH)
chat_service = ChatService(llm, memory_service, prompts)


# =============================================================================
# Routes
# =============================================================================

@app.route('/')
def home():
    """Render main chat interface."""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint with SSE streaming."""
    data = request.json
    return Response(
        stream_with_context(
            chat_service.generate_response(
                message=data.get('message'),
                conversation_id=data.get('conversation_id'),
                mode=data.get('mode', 'normal'),
                history=data.get('history', [])
            )
        ),
        mimetype='text/event-stream'
    )


@app.route('/models', methods=['GET'])
def list_models():
    """List available Ollama models."""
    models = llm.list_available_models()
    current = llm.get_current_model()
    return jsonify({'models': models, 'current': current})


@app.route('/models/switch', methods=['POST'])
def switch_model():
    """Switch to a different chat model."""
    data = request.json
    model_name = data.get('model')
    if not model_name:
        return jsonify({'error': 'Model name required'}), 400

    success = llm.switch_model(model_name)
    if success:
        logger.info(f"Switched to model: {model_name}")
        return jsonify({'success': True, 'current': llm.get_current_model()})
    return jsonify({'error': 'Failed to switch model'}), 500


@app.route('/user-profile', methods=['GET'])
def get_user_profile():
    """Get user profile details."""
    profile = UserProfile.query.first()
    if not profile:
        return jsonify({'name': '', 'details': ''})
    return jsonify({'name': profile.name, 'details': profile.details})


@app.route('/user-profile', methods=['POST'])
def save_user_profile():
    """Save user profile details."""
    data = request.json
    profile = UserProfile.query.first()
    if not profile:
        profile = UserProfile()
        db.session.add(profile)
    profile.name = data.get('name', '')[:100]  # Limit name length
    profile.details = data.get('details', '')
    profile.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({'success': True})


@app.route('/token-usage', methods=['POST'])
def log_token_usage():
    """Log token usage after AI response."""
    data = request.json
    usage = TokenUsage(
        conversation_id=data.get('conversation_id'),
        tokens_output=data.get('tokens_output', 0),
        tokens_per_second=data.get('tokens_per_second', 0),
        duration_ms=data.get('duration_ms', 0)
    )
    db.session.add(usage)
    db.session.commit()

    # Also log to file/console
    log_token(
        logger,
        usage.tokens_output,
        usage.tokens_per_second,
        usage.duration_ms,
        usage.conversation_id
    )

    return jsonify({'success': True, 'id': usage.id})


@app.route('/usage-stats', methods=['GET'])
def get_usage_stats():
    """Get global usage statistics."""
    # Total tokens all-time
    total_tokens = db.session.query(func.sum(TokenUsage.tokens_output)).scalar() or 0

    # Total conversations (saved)
    total_conversations = Conversation.query.count()

    # Average tokens per chat (only for saved conversations with usage data)
    avg_result = db.session.query(
        func.avg(TokenUsage.tokens_output)
    ).filter(TokenUsage.conversation_id.isnot(None)).scalar()
    avg_per_chat = round(avg_result or 0)

    # Average tokens per second
    avg_speed = db.session.query(
        func.avg(TokenUsage.tokens_per_second)
    ).filter(TokenUsage.tokens_per_second > 0).scalar()
    avg_tokens_per_second = round(avg_speed or 0, 1)

    return jsonify({
        'total_tokens': total_tokens,
        'total_conversations': total_conversations,
        'avg_per_chat': avg_per_chat,
        'avg_tokens_per_second': avg_tokens_per_second
    })


@app.route('/semantic-memory-stats', methods=['GET'])
def get_semantic_memory_stats():
    """Get semantic memory (Tier 2) statistics."""
    return jsonify(memory_service.get_stats())


@app.route('/conversations', methods=['GET'])
def get_conversations():
    """Get all saved conversations."""
    convos = Conversation.query.order_by(Conversation.last_updated.desc()).all()
    return jsonify([{
        'id': c.id,
        'title': c.title,
        'summary': c.summary,
        'last_updated': c.last_updated.isoformat()
    } for c in convos])


@app.route('/conversation/<int:id>', methods=['GET'])
def get_conversation(id):
    """Load a conversation with its messages."""
    convo = Conversation.query.get_or_404(id)
    messages = [{'role': m.role, 'content': m.content} for m in convo.messages]
    return jsonify({
        'id': convo.id,
        'title': convo.title,
        'summary': convo.summary,
        'messages': messages
    })


@app.route('/conversation/<int:id>', methods=['DELETE'])
def delete_conversation(id):
    """Delete a conversation."""
    convo = Conversation.query.get_or_404(id)
    db.session.delete(convo)
    db.session.commit()
    return jsonify({'success': True})


@app.route('/save-chat', methods=['POST'])
def save_chat():
    """Save current conversation with AI-generated title/summary."""
    data = request.json
    messages = data.get('messages', [])

    result = chat_service.save_chat_with_metadata(messages)

    if 'error' in result:
        return jsonify(result), 400 if result['error'] == 'No messages to save' else 500

    return jsonify(result)


@app.route('/message', methods=['POST'])
def save_message():
    """Save a single message to a conversation."""
    data = request.json
    conversation_id = data.get('conversation_id')

    if not conversation_id:
        return jsonify({'error': 'conversation_id required'}), 400

    # Update conversation's last_updated
    convo = Conversation.query.get(conversation_id)
    if convo:
        convo.last_updated = datetime.utcnow()

    msg = Message(
        conversation_id=conversation_id,
        role=data['role'],
        content=data['content']
    )
    db.session.add(msg)
    db.session.commit()

    # Trigger auto-summarization after assistant messages
    summarized = False
    if data['role'] == 'assistant':
        summarized = chat_service.auto_summarize(conversation_id)

    return jsonify({'id': msg.id, 'summarized': summarized})


@app.route('/memory-stats/<int:conversation_id>', methods=['GET'])
def get_memory_stats(conversation_id):
    """Get simplified memory statistics for a conversation."""
    convo = Conversation.query.get(conversation_id)
    if not convo:
        return jsonify({'error': 'Not found'}), 404

    total = Message.query.filter_by(conversation_id=conversation_id).count()

    return jsonify({
        'total_messages': total,
        'messages_summarized': convo.messages_summarized or 0,
        'has_summary': bool(convo.rolling_summary),
        'last_summary_at': convo.last_summary_at.isoformat() if convo.last_summary_at else None
    })


if __name__ == '__main__':
    logger.info(f"Starting Joey-Bot on port {config.PORT}")
    app.run(debug=config.DEBUG, port=config.PORT)
