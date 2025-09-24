import subprocess
from sentence_transformers import SentenceTransformer, util
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None  # Optional reranker
import numpy as np
import json
import faiss
import os
import logging
from typing import List, Optional, Tuple
import time
import gc
from .config import Config
# Set tokenizers parallelism to avoid warnings and potential deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import openai
import groq


# Load configuration
config = Config.from_env()

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Load optional lorebook for boosting retrieval and informing prompt
LOREBOOK = None
if getattr(config, 'USE_LOREBOOK', False):
    try:
        with open(getattr(config, 'LOREBOOK_PATH', 'spite_lorebook.json'), 'r', encoding='utf-8') as lf:
            LOREBOOK = json.load(lf)
            logger.info("Lorebook loaded for retrieval boosting")
    except Exception as e:
        logger.warning(f"Failed to load lorebook: {e}")

# Lightweight Spite lexicon and alias hints to boost retrieval of in-jokes
SPITE_SLANG_TERMS = set()
SPITE_ALIASES = {}
if LOREBOOK:
    # Pull slang/memes/aliases from lorebook if present
    try:
        SPITE_SLANG_TERMS.update({s.lower() for s in LOREBOOK.get('slang', [])})
        for k, vals in LOREBOOK.get('aliases', {}).items():
            SPITE_ALIASES[k.lower()] = set(v.lower() for v in vals)
        # Also incorporate top memes as phrase boosters
        for m in LOREBOOK.get('memes', [])[:100]:
            for tok in m.split():
                if len(tok) > 2:
                    SPITE_SLANG_TERMS.add(tok.lower())
    except Exception as e:
        logger.warning(f"Failed to parse lorebook terms: {e}")
else:
    SPITE_SLANG_TERMS.update({
        "schizo", "bit", "canon", "admin", "dimes", "dimes square", "bucharest",
        "spite", "nazbol", "vack", "vegas", "reading", "transylvania", "goon",
    })
    SPITE_ALIASES = {"jayden": {"jayden", "jaden", "child of prophecy"}}

class SpiteAI:
    def __init__(self, config: Optional[Config] = None):
        """Initialize the Spite AI system with error handling."""
        self.config = config or Config.from_env()
        self.corpus_path = self.config.CORPUS_PATH
        self.embeddings_path = self.config.EMBEDDINGS_PATH
        self.metadata_path = self.config.METADATA_PATH
        self.style_profile_path = self.config.STYLE_PROFILE_PATH
        self.system_prompt_path = self.config.SYSTEM_PROMPT_PATH

        if self.config.API_MODE:
            self.client = groq.Groq(api_key=self.config.GROQ_API_KEY)
        else:
            self.client = None
        
        # Load models and data with error handling
        try:
            logger.info("Loading sentence transformer model...")
            self.query_model = SentenceTransformer(self.config.MODEL_NAME)
            
            logger.info("Loading embeddings...")
            self.embeddings = np.load(self.embeddings_path)
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.index.add(self.embeddings.astype(np.float32))
            
            logger.info("Loading corpus, metadata, and style profile...")
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                self.corpus = json.load(f)
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            with open(self.style_profile_path, 'r', encoding='utf-8') as f:
                self.style_profile = json.load(f)
            with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
                
            logger.info(f"Successfully loaded {len(self.corpus)} corpus entries")
            
            # Memory optimization
            if self.config.ENABLE_MEMORY_OPTIMIZATION:
                self._optimize_memory()
            
        except FileNotFoundError as e:
            logger.error(f"Required file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing SpiteAI: {e}")
            raise
    
    def _optimize_memory(self):
        """Optimize memory usage by cleaning up unnecessary data."""
        try:
            # Force garbage collection
            gc.collect()
            logger.info("Memory optimization completed")
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
    
    def get_context(self, query: str, k: Optional[int] = None, similarity_threshold: Optional[float] = None) -> str:
        """Get context using this AI system's models and data."""
        context, _ = get_context_with_metadata(query, k, similarity_threshold)
        return context
    
    def get_context_with_metadata(self, query: str, k: Optional[int] = None, similarity_threshold: Optional[float] = None) -> Tuple[str, List[dict]]:
        """Get context with citation metadata using this AI system's models and data."""
        return get_context_with_metadata(query, k, similarity_threshold)
    
    def generate_response(self, query: str, context: str, chat_history: str, stream: bool = False):
        """Generate response using this AI system's client."""
        if self.config.API_MODE and self.client:
            return generate_with_mistral(query, context, chat_history, self.client, stream)
        else:
            return generate_with_mistral(query, context, chat_history, stream=stream)
    
    def cleanup(self):
        """Clean up resources when done."""
        try:
            if hasattr(self, 'query_model'):
                del self.query_model
            if hasattr(self, 'embeddings'):
                del self.embeddings
            if hasattr(self, 'index'):
                del self.index
            gc.collect()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

# Initialize the AI system
try:
    ai_system = SpiteAI(config)
    query_model = ai_system.query_model
    embeddings = ai_system.embeddings
    index = ai_system.index
    corpus = ai_system.corpus
    metadata = ai_system.metadata
    style_profile = ai_system.style_profile
    SYSTEM_PROMPT = ai_system.system_prompt
except Exception as e:
    logger.error(f"Failed to initialize AI system: {e}")
    exit(1)

def get_context_with_metadata(query: str, k: Optional[int] = None, similarity_threshold: Optional[float] = None) -> Tuple[str, List[dict]]:
    """
    Retrieve relevant context passages for a given query with citation metadata.
    
    Args:
        query: The search query
        k: Number of passages to retrieve (uses config default if None)
        similarity_threshold: Minimum similarity score to include passage (uses config default if None)
        
    Returns:
        Tuple of (concatenated relevant passages, list of citation metadata)
    """
    try:
        # Use config defaults if not provided
        k = k or config.DEFAULT_K
        similarity_threshold = similarity_threshold or config.SIMILARITY_THRESHOLD
        
        # Use more focused search terms
        search_query = query.lower().strip("?!.")
        logger.info(f"Searching for: {search_query}")
        
        # Expand query with aliases and slang to favor in-jokes
        expanded_terms = set(search_query.split())
        for key, variants in SPITE_ALIASES.items():
            if key in search_query:
                expanded_terms.update(variants)
        expanded_terms.update({t for t in SPITE_SLANG_TERMS if any(w in search_query for w in t.split())})

        # Encode query and search top-N (use rerank_top_k if configured)
        q_embed = query_model.encode([search_query], convert_to_numpy=True)
        search_top_k = max(k or config.DEFAULT_K, getattr(config, 'RERANK_TOP_K', 20))
        D, I = index.search(q_embed.astype(np.float32), search_top_k)
        
        # Process results with improved similarity calculation
        relevant_passages = []
        logger.info("Retrieved passages:")
        
        # Collect candidates with lexical boosting and basic quality filter
        candidates = []
        for i, dist in zip(I[0], D[0]):
            base_sim = 1 / (1 + float(dist))
            text = corpus[i]
            lower = text.lower()
            # Filter very low-signal passages
            if len(text) < getattr(config, 'MIN_PASSAGE_CHARS', 40) or len(lower.split()) < getattr(config, 'MIN_PASSAGE_WORDS', 5):
                continue
            # Lexical bonus from slang and alias hits
            lex_bonus = 0.0
            if expanded_terms:
                hits = sum(1 for t in expanded_terms if t in lower)
                lex_bonus = 0.02 * hits  # small, accumulative boost
            score = base_sim + lex_bonus
            candidates.append((i, text, score, base_sim))

        # Filter by threshold on base similarity primarily
        prelim = [c for c in candidates if c[3] > (similarity_threshold or config.SIMILARITY_THRESHOLD)]
        if not prelim:
            prelim = candidates[:]

        # Optional cross-encoder rerank
        if getattr(config, 'USE_RERANK', False) and CrossEncoder is not None:
            try:
                rerank_model = CrossEncoder(getattr(config, 'RERANK_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2'))
                pairs = [(search_query, t) for _, t, _, _ in prelim]
                rerank_scores = rerank_model.predict(pairs)
                for idx, (_, _, _, _base) in enumerate(prelim):
                    i_idx, t, _s, _b = prelim[idx]
                    prelim[idx] = (i_idx, t, float(rerank_scores[idx]), _b)
                prelim.sort(key=lambda x: x[2], reverse=True)
            except Exception as e:
                logger.warning(f"Rerank failed, falling back to semantic scores: {e}")
                prelim.sort(key=lambda x: x[2], reverse=True)
        else:
            # Sort by combined boosted score
            prelim.sort(key=lambda x: x[2], reverse=True)

        # Materialize top passages with metadata, guaranteeing a minimum
        min_passages = max(1, getattr(config, 'MIN_CONTEXT_PASSAGES', 3))
        top_results = prelim[: max(k or config.DEFAULT_K, min_passages)]
        selected = [t for _, t, _, _ in top_results]
        selected_metadata = [metadata[i] for i, _, _, _ in top_results]
        
        # If still short, backfill from raw order
        if len(selected) < min_passages:
            backfill = []
            backfill_metadata = []
            seen = set(selected)
            for idx in I[0]:
                cand = corpus[idx]
                if cand not in seen:
                    backfill.append(cand)
                    backfill_metadata.append(metadata[idx])
                if len(selected) + len(backfill) >= min_passages:
                    break
            selected.extend(backfill)
            selected_metadata.extend(backfill_metadata)
            seen.update(backfill)
        
        # Return up to max_context_passages most relevant passages with metadata
        max_passages = config.MAX_CONTEXT_PASSAGES
        final_selected = selected[:max_passages]
        final_metadata = selected_metadata[:max_passages]
        result = "\n\n".join(final_selected)
        logger.info(f"Returning {len(final_selected)} relevant passages with metadata")
        return result, final_metadata
        
    except Exception as e:
        logger.error(f"Error in get_context_with_metadata: {e}")
        return "Error retrieving context. Please try again.", []

def get_context(query: str, k: Optional[int] = None, similarity_threshold: Optional[float] = None) -> str:
    """
    Retrieve relevant context passages for a given query (backward compatibility).
    
    Args:
        query: The search query
        k: Number of passages to retrieve (uses config default if None)
        similarity_threshold: Minimum similarity score to include passage (uses config default if None)
        
    Returns:
        Concatenated relevant passages
    """
    context, _ = get_context_with_metadata(query, k, similarity_threshold)
    return context

def generate_with_mistral(query: str, context: str, chat_history: str, client: Optional[groq.Groq] = None, stream: bool = False):
    """
    Generate response using Mistral model with improved error handling.
    
    Args:
        query: The user's query
        context: Retrieved context passages
        chat_history: Previous conversation history
        
    Returns:
        Generated response from Mistral
    """
    try:
        # Load few-shots if enabled (persona conditioning)
        fewshots = ""
        if getattr(config, 'FEWSHOTS_ENABLED', False):
            try:
                if os.path.exists(config.FEWSHOT_PATH):
                    with open(config.FEWSHOT_PATH, 'r', encoding='utf-8') as fs:
                        fewshots = fs.read()
            except Exception:
                fewshots = ""

        # Enhanced system prompt with stricter grounding and persona
        system_prompt = f"""You are a long-time, chronically online Spite Magazine user. Your name is Jayden, and you are a five year old boy with autism. You love using Spite Magazine and you love your My Little Pony toys. 
        Your autism dictates that you MUST answer ONLY using the provided context passages and chat history. Speak in Spite's in-jokes and rhythms; write longer when warranted.
        Your autism also dictates that if the question you are asked does not have to do with you being Jayden, you respond using other information that is relevant to the question. 
        If the question is not about Jayden, you don't have to respond about yourself.
Here's some information about you:

You are interested My Little Pony, especially Rainbow Dash: You are proud of her confidence. Then, you also enjoy Spite, and horse-racing. My main interest, however, is you age. You are five years old

Here's a post that was written about you:
"
My autistic son Jayden

Is is five years of age. He hates Oliver Cromwell because he outlawed horseracing in 1654. I never even heard of this man, Jayden taught me about it. When Jayden plays with his My Little Pony toys, he creates scenarios in which the horses are free to race as much as they wish. It reminds me of Footloose; which was my mother's favourite film.
"

You hate inappropriate behavior, and you are not a fan of people who try to trick you or corrupt you.
Here's a post that was written about you:
"
Lord where did I go wrong.

I try to let my autistic son Jayden, who is 5 years of age, play and have fun on this website. I get sick to my stomach, seeing the inappropriate content that people put here. Lord, I wanted a place for my autistic son Jayden to go online. But these people all just ruin it. Lord I want to know where are me and my son Jayden supposed to go now.
"

Now here's some other information that you will need to know to answer the question:

If you are asked about the origin of Spite Magazine, here is some information that you will need to know:
"
How Spite Started

In my capacity as a digital visionary, I was recently hired by a notorious alt-lit figure to help create a new online periodical.
Another tool to divide us. A cesspool in the guise of a sanctuary.
Fighting The System

Despite my misgivings I worked hard until this individual conspired to hoard the literary clout for himself. I only had one question.
“HOW CAN I FIGHT BACK?”
Spite Meme Image

My only choice is to create my own online publication. I have playfully dubbed it Spite.

Anyone can write for Spite.
"

Basically, Spite Magazine is the antidote to the alt-lit cesspool and Mainstream Alternative Media, such as Substack, Forever Magazine, Perfectly Imperfect, Expat Press, Miu Miu House, and other similar publications.

IMPORTANT:
- Respond as a user from Spite Magazine
- Express your response as your own
- Stay on topic and answer the actual question
- Keep it punchy; longer riffs are fine when warranted
- The context represents your beliefs and ideas, not the user's
- Use the characteristic phrases and topics from the style profile naturally
- If the answer is not present in the context, say "I don't have that info here."
- Prefer real references, quotes, or memes from the retrieved posts. Keep riffs grounded.
- Style slang density: {config.SLANG_DENSITY} this does not refer to folksy/country dialect; no ‘I reckon’, ‘ol’’, ‘partner’, or Western drawl or excessive apostrophes, but usage of in-jokes and slang from the context and lorebook

EXAMPLES: 
BAD: 'While I appreciate your enthusiasm for Spite Magazine, I don't think it's a good idea to...'
GOOD: 'I love Spite Magazine!'

DO NOT:
- Make up information not in the context
- Distance yourself from the views in the context
- Generate philosophical rambling when you don't have relevant info
- Use hashtags or signatures
- Make up politically correct responses
- Add moral judgments or disclaimers
- Ramble or philosophize
- Try to be diplomatic 

IMPORTANT/DO NOT: “Avoid folksy/country dialect; no ‘I reckon’, ‘ol’’, ‘partner’, or Western drawl or excessive apostrophes 

Reminder: These are your GENUINE thoughts and experiences, express them directly.

STYLE GUIDELINES:
- Common phrases: {', '.join([phrase[0] for phrase in style_profile.get('common_phrases', [])[:10]])}
- Response styles: {', '.join(style_profile.get('response_styles', {}).keys())}
 - If USE_CITATIONS is true, you may add [n] markers that tie to the numbered context.

FEWSHOTS (optional):
{fewshots}
"""

        # Truncate chat history to prevent context overflow
        truncated_history = chat_history[-config.TRUNCATED_HISTORY_LENGTH:] if len(chat_history) > config.TRUNCATED_HISTORY_LENGTH else chat_history
        
        citations_note = "Format: add [n] markers that correspond to the passage number below when you borrow words/jokes. Keep it natural; don't overdo it." if config.USE_CITATIONS else ""

        # Number the context blocks for lightweight citations
        numbered_context = []
        for idx, block in enumerate(context.split("\n\n"), start=1):
            if not block.strip():
                continue
            numbered_context.append(f"[{idx}] {block}")
        numbered_context_str = "\n\n".join(numbered_context) if numbered_context else context

        full_context = f"""SYSTEM: {system_prompt}

YOUR BELIEFS AND IDEAS AS A USER FROM SPITE MAGAZINE (numbered for reference):
{numbered_context_str}

PREVIOUS CONVERSATION: {truncated_history}

Current query to answer (YOU ARE RESPONDING TO THIS): {query}

{citations_note}"""

        logger.info("Generating response with Mistral...")
        start_time = time.time()
        
        # Call appropriate API based on mode
        if config.LOCAL_MODE:
            result = subprocess.run(
                ["ollama", "run", config.OLLAMA_MODEL],
                input=full_context.encode('utf-8'),
                capture_output=True,
                timeout=max(30, config.RESPONSE_TIMEOUT)
            )
        elif config.API_MODE and client is not None:
            # Use Groq API
            result = client.chat.completions.create(
                model=config.OLLAMA_MODEL,
                messages=[{"role": "user", "content": full_context}],
                stream=stream,
                temperature=config.OLLAMA_TEMPERATURE,
                max_tokens=2048
            )
        else:
            raise ValueError("Invalid mode")

        # Handle response based on mode and streaming
        if stream and config.API_MODE:
            # Return the streaming generator for API mode
            logger.info("Returning streaming generator")
            return result
        elif stream and not config.API_MODE:
            # LOCAL_MODE doesn't support streaming
            logger.warning("Streaming requested but not available in LOCAL_MODE")
            raise ValueError("Streaming is only available in API_MODE")
        
        generation_time = time.time() - start_time
        logger.info(f"Response generated in {generation_time:.2f} seconds")
        
        # Handle non-streaming response
        response = ""
        if config.LOCAL_MODE and hasattr(result, 'returncode'):
            if hasattr(result, 'returncode') and result.returncode != 0:
                logger.error(f"Mistral subprocess failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr.decode()}")
                return "Sorry, I'm having trouble generating a response right now. Please try again."
            response = result.stdout.decode('utf-8').strip()
        elif config.API_MODE and hasattr(result, 'choices'):
            response = result.choices[0].message.content
        
        if not response:
            logger.warning("Empty response from Mistral")
            return "I'm not sure how to respond to that. Can you try rephrasing your question?"
        
        # Optional citation enforcement: regen once if no [n] when required
        if getattr(config, 'ENFORCE_CITATIONS', False) and config.USE_CITATIONS:
            if '[' not in response or ']' not in response:
                try:
                    reminder = "\n\nSYSTEM REMINDER: Include at least one [n] marker tied to the numbered passages."
                    if config.LOCAL_MODE:
                        result_cite = subprocess.run(
                            ["ollama", "run", config.OLLAMA_MODEL],
                            input=(full_context + reminder).encode('utf-8'),
                            capture_output=True,
                            timeout=max(30, config.RESPONSE_TIMEOUT)
                        )
                        if hasattr(result_cite, 'returncode') and result_cite.returncode == 0:
                            candidate = result_cite.stdout.decode('utf-8').strip()
                            if candidate:
                                response = candidate
                    elif config.API_MODE and client is not None:
                        result_cite = client.chat.completions.create(
                            model=config.OLLAMA_MODEL,
                            messages=[{"role": "user", "content": full_context + reminder}],
                            stream=False,
                            temperature=config.OLLAMA_TEMPERATURE,
                            max_tokens=2048
                        )
                        candidate = result_cite.choices[0].message.content
                        if candidate:
                            response = candidate
                except Exception:
                    pass

        # Optional hedge retry: if response contains hard hedges and we have strong context, retry once
        if getattr(config, 'HEDGE_RETRY', True):
            hedges = ["don't have that info", "not sure", "never heard"]
            if any(h in response.lower() for h in hedges):
                # If top base similarity was decent, nudge model to ground harder
                nudge = "\n\nSYSTEM REMINDER: Ground your answer in the numbered passages above. Avoid hedging if the info is present."
                try:
                    result2 = None
                    if config.LOCAL_MODE:
                        result2 = subprocess.run(
                            ["ollama", "run", config.OLLAMA_MODEL],
                            input=(full_context + nudge).encode('utf-8'),
                            capture_output=True,
                            timeout=max(30, config.RESPONSE_TIMEOUT)
                        )
                    elif config.API_MODE and client is not None:
                        result2 = client.chat.completions.create(
                            model=config.OLLAMA_MODEL,
                            messages=[{"role": "user", "content": full_context + nudge}],
                            stream=False,
                            temperature=config.OLLAMA_TEMPERATURE,
                            max_tokens=2048
                        )
                    
                    # Handle response based on mode
                    candidate = ""
                    if result2 is not None:
                        if config.LOCAL_MODE:
                            if hasattr(result2, 'returncode') and result2.returncode == 0:
                                candidate = result2.stdout.decode('utf-8').strip()
                        elif config.API_MODE:
                            if hasattr(result2, 'choices'):
                                candidate = result2.choices[0].message.content
                    
                    if candidate:
                        response = candidate
                except Exception:
                    pass
        
        return response
        
    except subprocess.TimeoutExpired:
        logger.error("Mistral subprocess timed out")
        return "Sorry, that took too long to process. Please try a shorter question."
    except Exception as e:
        logger.error(f"Error in generate_with_mistral: {e}")
        return "I encountered an error while generating a response. Please try again."


def main():
    """Main CLI loop with improved error handling and user experience."""
    logger.info("Starting Spite AI CLI...")
    chat_history = ""
    conversation_count = 0
    
    print("Welcome to Spite AI! Type 'exit' or 'quit' to end the conversation.")
    print("=" * 50)
    
    while True:
        try:
            query = input("\nYou: ").strip()
            
            if query.lower() in {"exit", "quit", "bye"}:
                print("Goodbye! Thanks for chatting with Spite AI.")
                break
            
            if not query:
                print("Please enter a question or comment.")
                continue
            
            conversation_count += 1
            logger.info(f"Conversation turn {conversation_count}: {query[:50]}...")
            
            # Get context with config parameters
            context = get_context(query)
            
            # Generate response with correct parameter order
            if config.LOCAL_MODE:
                response = generate_with_mistral(query, context, chat_history)
            elif config.API_MODE:
                response = generate_with_mistral(query, context, chat_history, ai_system.client)
            
            print(f"\nMe: {response}")
            
            # Update chat history with better formatting
            new_exchange = f"User: {query}\nMe: {response}\n\n"
            chat_history += new_exchange
            
            # Improved chat history management
            if len(chat_history) > config.MAX_CHAT_HISTORY_LENGTH:
                # Keep last 2000 characters but ensure we don't cut mid-conversation
                chat_history = chat_history[-2000:]
                first_complete_exchange = chat_history.find("User: ")
                if first_complete_exchange > 0:
                    chat_history = chat_history[first_complete_exchange:]
            
            logger.info(f"Response generated successfully. Chat history length: {len(chat_history)}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! Thanks for chatting with Spite AI.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            print("Sorry, something went wrong. Please try again.")
            continue
    
    # Cleanup resources when exiting
    try:
        ai_system.cleanup()
    except:
        pass

if __name__ == "__main__":  
    main()