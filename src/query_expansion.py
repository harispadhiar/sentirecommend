"""
query_expansion.py — Query expansion lexicon & utilities
Maps user queries to related terms for better semantic matching.

When user searches "presentation", expand to include:
  - "slide", "deck", "slideshow", "powerpoint", "keynote"

This compensates for literal TF-IDF matching that misses synonyms.
"""

# Define synonym groups — each key is the primary term, values are related terms
QUERY_EXPANSION_MAP = {
    # Presentations & slides
    "presentation": ["slide", "deck", "slideshow", "powerpoint", "keynote", "presentation software"],
    "slide": ["presentation", "deck", "slideshow", "powerpoint", "keynote"],
    "deck": ["presentation", "slide", "slideshow", "powerpoint"],
    "slideshow": ["presentation", "slide", "deck", "powerpoint"],
    
    # Writing & documents
    "writing": ["document", "writer", "text", "content", "grammar", "spell check"],
    "document": ["writing", "writer", "text", "pdf", "content"],
    "email": ["mail", "message", "communication", "compose", "draft"],
    
    # Video & multimedia
    "video": ["youtube", "editing", "film", "movie", "motion", "animation", "footage"],
    "editing": ["video", "film", "movie", "motion", "audio", "post-production"],
    "animation": ["video", "motion", "film", "design", "graphics"],
    
    # Design & graphics
    "design": ["graphic", "creative", "ui", "ux", "visual", "art", "branding", "poster"],
    "graphic": ["design", "visual", "art", "creative", "ui", "poster", "image"],
    "poster": ["design", "graphic", "visual", "branding", "template"],
    
    # Code & development
    "coding": ["code", "programming", "developer", "software", "debug", "algorithm"],
    "code": ["coding", "programming", "developer", "software", "script", "debug"],
    "programming": ["code", "coding", "developer", "software", "algorithm"],
    
    # Chat & conversation
    "chatbot": ["chat", "conversation", "messaging", "ai assistant", "dialogue"],
    "chat": ["chatbot", "conversation", "messaging", "dialogue", "communication"],
    "conversation": ["chat", "dialogue", "messaging", "communication"],
    
    # Image & photos
    "image": ["photo", "picture", "visual", "graphic", "artwork", "illustration"],
    "photo": ["image", "picture", "visual", "photography", "picture editing"],
    "picture": ["image", "photo", "visual", "graphic"],
    
    # Productivity
    "productivity": ["task", "project", "management", "organization", "planning", "time"],
    "task": ["productivity", "todo", "project", "management", "planning"],
    "project": ["management", "productivity", "planning", "task", "team"],
    
    # Marketing & business
    "marketing": ["seo", "social", "advertising", "campaign", "brand", "promotion"],
    "seo": ["marketing", "search", "optimization", "ranking", "traffic"],
    "social": ["marketing", "media", "advertising", "posts", "content"],
    
    # Transcription & speech
    "transcription": ["transcribe", "speech", "audio", "voice", "subtitle"],
    "transcribe": ["transcription", "speech", "audio", "voice"],
    "voice": ["audio", "speech", "transcription", "sound", "podcast"],
    
    # Summarization & extraction
    "summarize": ["summary", "abstract", "condense", "digest", "extract", "compress"],
    "summary": ["summarize", "abstract", "digest", "extract"],
    
    # Search & discovery
    "search": ["find", "query", "lookup", "discovery", "retrieval"],
    
    # AI & machine learning
    "ai": ["artificial intelligence", "machine learning", "automation", "algorithm"],
    "automation": ["ai", "automation", "workflow", "process", "scheduling"],
}


def expand_query(query_text):
    """
    Expand a user query with related synonyms.
    
    Parameters
    ----------
    query_text : str
        User's search query
    
    Returns
    -------
    str
        Expanded query with synonyms added
        Example: "presentation" → "presentation slide deck slideshow"
    """
    if not query_text or not query_text.strip():
        return query_text
    
    query_lower = query_text.lower().strip()
    words = query_lower.split()
    
    expanded_terms = set(words)  # Start with original words
    
    # For each word, add its expansions
    for word in words:
        if word in QUERY_EXPANSION_MAP:
            expanded_terms.update(QUERY_EXPANSION_MAP[word])
    
    # Return expanded query as space-separated string
    return " ".join(sorted(expanded_terms))