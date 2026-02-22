"""Flask application entry point for Joey-Bot."""

from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from sqlalchemy import func

from app.data.database import db, Conversation, Message, UserProfile, TokenUsage, init_db
from app.models.ollama_wrapper import OllamaWrapper
from app.services.chat_service import ChatService
from app.services.memory_service import MemoryService
from app.services.gatekeeper import MemoryGatekeeper
from app.services.orchestrator import ChatOrchestrator
from app.services.reranker import HeuristicReranker
from app.services.memory_lifecycle import MemoryLifecycle, get_last_lifecycle_run
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
gatekeeper = MemoryGatekeeper(
    llm, prompts,
    max_tokens=config.GATEKEEPER_MAX_TOKENS,
    timeout=config.GATEKEEPER_TIMEOUT
)
chat_service = ChatService(llm, memory_service, prompts)
reranker = HeuristicReranker()
orchestrator = ChatOrchestrator(
    gatekeeper=gatekeeper,
    retriever=memory_service.retriever,
    chat_service=chat_service,
    memory_service=memory_service,
    reranker=reranker,
)
lifecycle = MemoryLifecycle(store=memory_service.store, memory_service=memory_service)

# Pipeline run history (in-memory, not persisted)
_pipeline_runs: list = []
MAX_PIPELINE_HISTORY = 10


def _record_pipeline_run(orch):
    """Extract diagnostics from the last pipeline context and store in history."""
    ctx = orch._last_pipeline_ctx
    if not ctx:
        return

    timings = ctx.get("timings", {})
    errors = ctx.get("errors", [])
    classification = ctx.get("classification", {})
    memory_need = classification.get("memory_need", "SEMANTIC")
    candidates = ctx.get("scored_candidates", [])

    # Build per-stage status/detail
    error_stages = {e["stage"].lower() for e in errors}
    skip_stages = set()
    if memory_need in ("NONE", "RECENT", "PROFILE"):
        skip_stages.update(("retrieve", "score"))

    stages = {}
    stage_names = ["classify", "retrieve", "score", "build_context", "generate", "post_process"]
    for name in stage_names:
        if name in error_stages:
            err = next((e for e in errors if e["stage"].lower() == name), {})
            stages[name] = {
                "status": "error",
                "detail": f"{err.get('error', 'unknown')} → {err.get('fallback', '')}",
            }
        elif name in skip_stages:
            stages[name] = {"status": "skipped", "detail": f"memory_need={memory_need}"}
        else:
            stages[name] = {"status": "success", "detail": ""}

    # Fill in detail text for successful stages
    if stages["classify"]["status"] == "success":
        conf = classification.get("confidence", 0)
        stages["classify"]["detail"] = f"{memory_need} (conf={conf:.2f})"

    if stages["retrieve"]["status"] == "success":
        stages["retrieve"]["detail"] = f"{len(ctx.get('candidates', []))} candidates"

    if stages["score"]["status"] == "success":
        before = len(ctx.get("candidates", []))
        after = len(candidates)
        stages["score"]["detail"] = f"{before} -> {after} reranked"

    if stages["build_context"]["status"] == "success":
        stages["build_context"]["detail"] = f"{len(ctx.get('assembled_prompt', ''))} chars"

    if stages["generate"]["status"] == "success":
        response = ctx.get("response_text", "")
        approx_tokens = len(response.split())
        stages["generate"]["detail"] = f"~{approx_tokens} tokens"

    if stages["post_process"]["status"] == "success":
        stages["post_process"]["detail"] = "completed"

    run = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_message": (ctx.get("user_message", "") or "")[:120],
        "classification": {
            "memory_need": memory_need,
            "retrieval_keys": classification.get("retrieval_keys", []),
            "confidence": classification.get("confidence", 0),
        },
        "candidate_count": len(candidates),
        "total_time_ms": round(sum(timings.values()), 1),
        "timings": {k: round(v, 1) for k, v in timings.items()},
        "errors": errors,
        "stages": stages,
        "prompt_length": len(ctx.get("assembled_prompt", "")),
    }

    _pipeline_runs.insert(0, run)
    del _pipeline_runs[MAX_PIPELINE_HISTORY:]


# =============================================================================
# Background scheduler (memory lifecycle)
# =============================================================================

def _run_lifecycle_task(task_name, task_func):
    """Run a lifecycle task inside the Flask app context."""
    with app.app_context():
        logger.info(f"[SCHEDULER] Running {task_name}")
        try:
            result = task_func()
            logger.info(f"[SCHEDULER] {task_name} complete: {result}")
        except Exception as e:
            logger.warning(f"[SCHEDULER] {task_name} error: {e}")

if config.LIFECYCLE_ENABLED:
    from apscheduler.schedulers.background import BackgroundScheduler
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(
        lambda: _run_lifecycle_task("decay", lifecycle.decay_all_strengths),
        "interval", hours=config.LIFECYCLE_DECAY_HOURS, id="lifecycle_decay",
    )
    scheduler.add_job(
        lambda: _run_lifecycle_task("consolidate", lifecycle.consolidate),
        "interval", hours=config.LIFECYCLE_CONSOLIDATE_HOURS, id="lifecycle_consolidate",
    )
    scheduler.add_job(
        lambda: _run_lifecycle_task("prune", lifecycle.prune),
        "interval", days=config.LIFECYCLE_PRUNE_DAYS, id="lifecycle_prune",
    )
    scheduler.start()
    logger.info(
        f"[SCHEDULER] Memory lifecycle enabled: "
        f"decay={config.LIFECYCLE_DECAY_HOURS}h, "
        f"consolidate={config.LIFECYCLE_CONSOLIDATE_HOURS}h, "
        f"prune={config.LIFECYCLE_PRUNE_DAYS}d"
    )


# =============================================================================
# Routes
# =============================================================================

@app.route('/')
def home():
    """Render main chat interface."""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint with SSE streaming via orchestrator pipeline."""
    data = request.json
    conversation_id = data.get('conversation_id')
    history = data.get('history', [])

    # Load recent messages from DB or in-memory history
    recent_messages = []
    if conversation_id:
        recent_db_msgs = Message.query.filter_by(conversation_id=conversation_id)\
            .order_by(Message.timestamp.desc()).limit(config.MAX_RECENT_MESSAGES).all()
        recent_db_msgs.reverse()
        recent_messages = [
            {"role": msg.role, "content": msg.content} for msg in recent_db_msgs
        ]
    elif history:
        recent_messages = history[-config.MAX_RECENT_MESSAGES:]

    def generate_and_record():
        yield from orchestrator.process_message(
            user_message=data.get('message'),
            conversation_id=conversation_id,
            recent_messages=recent_messages,
            mode=data.get('mode', 'normal'),
        )
        _record_pipeline_run(orchestrator)

    return Response(
        stream_with_context(generate_and_record()),
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
    """Get semantic memory (Tier 2) statistics with strength tiers."""
    stats = memory_service.get_stats()

    # Add strength tier counts and consolidated count
    if stats.get('total_facts', 0) > 0:
        all_memories = memory_service.store.get_all()
        metadatas = all_memories.get("metadatas", [])

        strong = medium = weak = consolidated_count = 0
        for meta in metadatas:
            s = float(meta.get("strength", 0.5))
            if s > 0.7:
                strong += 1
            elif s >= 0.3:
                medium += 1
            else:
                weak += 1
            if meta.get("consolidated"):
                consolidated_count += 1

        stats["strength_tiers"] = {
            "strong": strong,
            "medium": medium,
            "weak": weak,
        }
        stats["consolidated_count"] = consolidated_count

    stats["last_lifecycle_run"] = get_last_lifecycle_run()
    return jsonify(stats)


@app.route('/memory-lifecycle', methods=['POST'])
def run_memory_lifecycle():
    """Manual trigger for memory lifecycle operations."""
    data = request.json or {}
    action = data.get('action', 'full')

    actions = {
        'decay': lifecycle.decay_all_strengths,
        'consolidate': lifecycle.consolidate,
        'prune': lifecycle.prune,
        'full': lifecycle.run_full_cycle,
    }

    if action not in actions:
        return jsonify({'error': f'Unknown action: {action}. Use: {list(actions.keys())}'}), 400

    try:
        result = actions[action]()
        return jsonify({'action': action, 'result': result})
    except Exception as e:
        logger.warning(f"[LIFECYCLE] Manual {action} error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/dashboard')
def dashboard():
    """Render pipeline observability dashboard."""
    return render_template('dashboard.html')


@app.route('/api/pipeline-runs', methods=['GET'])
def get_pipeline_runs():
    """Get recent pipeline run history (up to 10)."""
    return jsonify(_pipeline_runs)


@app.route('/api/pipeline-latest', methods=['GET'])
def get_pipeline_latest():
    """Get the most recent pipeline run."""
    return jsonify(_pipeline_runs[0] if _pipeline_runs else None)


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
