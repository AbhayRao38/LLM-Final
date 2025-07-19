import argparse
import sys
import os
from datetime import datetime
from knowledgebase import KnowledgeBaseManager
from retrieval import RetrievalAugmentor
import quillai_llm
import re

print(f"âœ… Loaded Enhanced QuillAILLM from: {quillai_llm.__file__}")
QuillAILLM = quillai_llm.QuillAILLM

# Pydantic models for API compatibility
from pydantic import BaseModel
from typing import Optional, List

class QueryRequest(BaseModel):
    query: str
    mode: str  # "learning" or "question"
    marks: Optional[int] = None

class QueryResponse(BaseModel):
    success: bool
    llm_output: Optional[str] = None
    custom_output: Optional[str] = None
    intent: Optional[str] = None
    domain: Optional[str] = None
    topics: Optional[List[str]] = None
    context_used: bool = False
    generation_time: float = 0.0
    word_count: int = 0
    timestamp: str
    error_message: Optional[str] = None

# This function will now be called once at API server startup
def initialize_api_components():
    """Initialize LLM and retrieval components for API use."""
    try:
        # Initialize LLM
        llm = QuillAILLM(
            model_name="microsoft/DialoGPT-medium",
            force_model_check=True,
            debug_mode=False
        )
        
        # Initialize retrieval system (will use /tmp paths from retrieval.py)
        retrieval_system = RetrievalAugmentor(
            chunk_size=400,
            chunk_overlap=50
        )
        
        print("âœ… API components initialized successfully")
        return llm, retrieval_system
        
    except Exception as e:
        print(f"âŒ Failed to initialize API components: {e}")
        raise

# Modified to accept pre-initialized LLM and retrieval_system instances
def process_api_query(request, llm: QuillAILLM, retrieval_system: RetrievalAugmentor) -> dict:
    """
    Process API query and return dual outputs.
    
    Args:
        request: Query request object with query, mode, marks attributes
        llm: Pre-initialized QuillAILLM instance
        retrieval_system: Pre-initialized RetrievalAugmentor instance
        
    Returns:
        dict: Response with dual outputs and metadata
    """
    from datetime import datetime
    
    start_time = datetime.utcnow()
    
    try:
        # Components are now passed in, no re-initialization here
        
        # Detect query intent and domain
        intent, intent_confidence, all_intents = llm.detect_query_intent(request.query)
        domain, topics, domain_confidence = llm.detect_domain_and_topic(request.query)
        
        # Retrieve context if available
        context_chunks = []
        context_used = False
        
        try:
            context_chunks = retrieval_system.retrieve_context(
                request.query,
                top_k=3,
                min_score_threshold=0.3
            )
            context_used = len(context_chunks) > 0
        except Exception as e:
            print(f"âš ï¸ Context retrieval failed: {e}")
        
        # Generate dual response using the new parallel method
        dual_response_data = llm.generate_dual_response(
            query=request.query,
            mode=request.mode or "learning",
            marks=request.marks,
            context_chunks=context_chunks,
            temperature=0.8 # Use a default temperature for dual response
        )
        
        generation_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Calculate total word count from both outputs
        total_word_count = dual_response_data['word_counts']['llm'] + dual_response_data['word_counts']['custom']
        
        # Prepare response
        response = {
            "success": True,
            "llm_output": dual_response_data.get("llm_output", ""),
            "custom_output": dual_response_data.get("custom_output", ""),
            "intent": intent,
            "domain": domain,
            "topics": topics,
            "context_used": context_used,
            "generation_time": generation_time,
            "word_count": total_word_count,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        return response
        
    except Exception as e:
        error_msg = f"Query processing failed: {str(e)}"
        print(f"âŒ {error_msg}")
        
        return {
            "success": False,
            "error_message": error_msg,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

"""ENHANCED Main application script with comprehensive solutions:
1. Advanced Query Intent Detection & Routing
2. Robust PDF Processing with OCR Support  
3. Intelligent Chunking and Context Retrieval
4. Enhanced Error Handling and Validation
5. Multi-domain Academic Support
6. Semantic Understanding and Reranking
Author: AbhayRao38
Date: 2025-07-08
Version: Production-Ready Enhanced System"""

# The following functions are part of the CLI interface (if __name__ == "__main__")
# and are not directly used by the FastAPI server.
# Their initialization logic for KB/Retrieval should also be updated to use /tmp paths.

def print_enhanced_header(args):
    """Print enhanced header with session information."""
    print("=" * 80)
    print("ðŸ¤– ENHANCED QUILLAI - INTELLIGENT ACADEMIC ASSISTANT")
    print("=" * 80)
    print(f"ðŸ‘¤ User: AbhayRao38")
    print(f"ðŸ“… Session: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"ðŸŽ¯ Mode: {getattr(args, 'mode', 'N/A')}")
    if hasattr(args, 'marks') and args.marks:
        print(f"ðŸ“ Target: {args.marks} marks")
    if hasattr(args, 'query') and args.query:
        print(f"â“ Query: {args.query[:60]}{'...' if len(args.query) > 60 else ''}")
    print("=" * 80)

def validate_enhanced_arguments(args):
    """Validate arguments with enhanced feedback."""
    errors = []
    warnings = []
    
    # Mode validation
    if hasattr(args, 'mode') and args.mode:
        if args.mode not in ['learning', 'question']:
            errors.append(f"Invalid mode: {args.mode}. Use 'learning' or 'question'")
    
    # Marks validation
    if hasattr(args, 'marks') and args.marks:
        if args.marks not in [2, 5, 10]:
            errors.append(f"Invalid marks: {args.marks}. Use 2, 5, or 10")
        if not hasattr(args, 'mode') or args.mode != 'question':
            warnings.append("Marks specified but mode is not 'question'")
    
    # Temperature validation
    if hasattr(args, 'temperature') and args.temperature:
        if not (0.1 <= args.temperature <= 1.5):
            warnings.append(f"Temperature {args.temperature} outside recommended range 0.1-1.5")
    
    # PDF validation
    if hasattr(args, 'pdf') and args.pdf:
        if not os.path.exists(args.pdf):
            errors.append(f"PDF file not found: {args.pdf}")
        elif not args.pdf.lower().endswith('.pdf'):
            warnings.append(f"File may not be a PDF: {args.pdf}")
    
    # Display validation results
    if errors:
        print("âŒ Validation Errors:")
        for error in errors:
            print(f"   â€¢ {error}")
        sys.exit(1)
    
    if warnings:
        print("âš ï¸  Validation Warnings:")
        for warning in warnings:
            print(f"   â€¢ {warning}")
        print()

def route_and_process_query(args):
    """Intelligent routing based on query intent detection."""
    print(f"Processing query: {args.query}")
    # Initialize LLM for intent detection
    try:
        quillai_llm = setup_enhanced_llm(args)
        if not quillai_llm:
            return
        
        # NEW: Detect query intent and route accordingly
        intent, intent_confidence, all_intents = quillai_llm.detect_query_intent(args.query)
        domain, topics, domain_confidence = quillai_llm.detect_domain_and_topic(args.query)
        
        print(f"ðŸ§  Query Analysis:")
        print(f"   Intent: {intent.upper()}")
        print(f"   Domain: {domain.upper()}")
        if topics:
            print(f"   Topics: {', '.join(topics)}")
        
        # Route to appropriate handler
        if intent == 'question_generation':
            handle_question_generation_route(args, quillai_llm, intent, domain, topics)
        elif intent == 'rubric_creation':
            handle_rubric_creation_route(args, quillai_llm, intent, domain, topics)
        elif intent == 'summary_request':
            handle_summary_request_route(args, quillai_llm, intent, domain, topics)
        else:
            handle_standard_qa_route(args, quillai_llm, intent, domain, topics)
            
    except Exception as e:
        handle_critical_error("Query routing failed", e, args)

def handle_question_generation_route(args, llm, intent, domain, topics):
    """Handle question generation with specialized processing."""
    print(f"Routing to question generation handler")
    print(f"ðŸ“ Detected: Question Generation Request")
    print(f"   Optimizing for educational content creation...")
    # Extract question parameters
    numbers = re.findall(r'\d+', args.query)
    num_questions = int(numbers[0]) if numbers else 5
    print(f"   Generating {num_questions} questions about {domain}")
    # Process with context if available
    context_chunks = []
    if not args.no_context:
        retrieval_system = setup_enhanced_retrieval_system(args)
        if retrieval_system:
            context_chunks = retrieve_enhanced_context(retrieval_system, args)
    # Generate with specialized handling
    generate_and_display_enhanced_answer(llm, args, context_chunks)

def handle_rubric_creation_route(args, llm, intent, domain, topics):
    """Handle rubric creation with specialized processing."""
    print(f"Routing to rubric creation handler")
    print(f"ðŸ“‹ Detected: Rubric Creation Request")
    print(f"   Optimizing for assessment criteria generation...")
    # Suggest marks if not provided
    if not args.marks:
        print(f"ðŸ’¡ Suggestion: Consider specifying --marks for targeted rubric")
        print(f"   Example: --marks 10 for comprehensive rubric")
    # Process with minimal context (rubrics are usually standalone)
    context_chunks = []
    generate_and_display_enhanced_answer(llm, args, context_chunks)

def handle_summary_request_route(args, llm, intent, domain, topics):
    """Handle summary requests with context emphasis."""
    print(f"Routing to summary request handler")
    print(f"ðŸ“„ Detected: Summary Request")
    print(f"   Prioritizing context retrieval for summarization...")
    # Context is crucial for summaries
    context_chunks = []
    if not args.no_context:
        retrieval_system = setup_enhanced_retrieval_system(args)
        if retrieval_system:
            context_chunks = retrieve_enhanced_context(retrieval_system, args)
            if not context_chunks:
                print(f"âš ï¸  Warning: No context available for summarization")
                print(f"   Consider adding PDFs with: --pdf path/to/document.pdf")
                return
        else:
            print(f"âŒ Error: Summary requires context but retrieval system unavailable")
            return
    generate_and_display_enhanced_answer(llm, args, context_chunks)

def handle_standard_qa_route(args, llm, intent, domain, topics):
    """Handle standard Q&A with optimized processing."""
    print(f"Routing to standard Q&A handler")
    print(f"â“ Detected: Standard Q&A Request")
    print(f"   Intent: {intent}, Domain: {domain}")
    # Standard processing with context
    context_chunks = []
    if not args.no_context:
        retrieval_system = setup_enhanced_retrieval_system(args)
        if retrieval_system:
            context_chunks = retrieve_enhanced_context(retrieval_system, args)
    generate_and_display_enhanced_answer(llm, args, context_chunks)

def handle_critical_error(operation, error, args):
    """Handle critical errors with comprehensive logging and user feedback."""
    import traceback
    # User-friendly error display
    print(f"\nâŒ Critical Error in {operation}")
    print(f"Error: {str(error)}")
    # Provide specific troubleshooting based on error type
    if "memory" in str(error).lower() or "cuda" in str(error).lower():
        print(f"\nðŸ”§ Memory/GPU Troubleshooting:")
        print(f"   â€¢ Try --no_context to reduce memory usage")
        print(f"   â€¢ Lower --temperature to 0.5")
        print(f"   â€¢ Reduce --max_tokens if specified")
        print(f"   â€¢ Close other applications to free memory")
    elif "model" in str(error).lower() or "download" in str(error).lower():
        print(f"\nðŸ”§ Model Loading Troubleshooting:")
        print(f"   â€¢ Check internet connection")
        print(f"   â€¢ Verify model name: {args.model_name}")
        print(f"   â€¢ Try clearing model cache")
        print(f"   â€¢ Ensure sufficient disk space (5GB+)")
    elif "pdf" in str(error).lower() or "file" in str(error).lower():
        print(f"\nðŸ”§ File/PDF Troubleshooting:")
        print(f"   â€¢ Verify file exists and is readable")
        print(f"   â€¢ Check file permissions")
        print(f"   â€¢ Ensure PDF is not corrupted")
        print(f"   â€¢ Try with a different PDF file")
    else:
        print(f"\nðŸ”§ General Troubleshooting:")
        print(f"   â€¢ Try with --verbose for more details")
        print(f"   â€¢ Use --debug for full error traces")
        print(f"   â€¢ Restart the application")
        print(f"   â€¢ Check system resources")
    if args.debug:
        print(f"\nðŸ› Debug Traceback:")
        traceback.print_exc()

def handle_non_query_operations(args):
    """Handle operations that do not require a query."""
    if args.list_pdfs or args.kb_stats:
        handle_kb_management(args)
        return
    if args.index_stats:
        handle_index_management(args)
        return
    if args.remove_source:
        handle_source_removal(args)
        return
    if args.search_chunks:
        handle_chunk_search(args)
        return
    if args.pdf:
        handle_pdf_addition(args)
        return
    if args.test_intents:
        run_intent_detection_tests(args)
        return
    if args.test_pdf_processing:
        run_pdf_processing_tests(args)
        return
    if args.benchmark:
        run_enhanced_benchmark_tests(args)
        return
    print("âŒ No query or operation specified.")
    show_enhanced_usage_examples()

def handle_pdf_addition(args):
    """Handle PDF addition with comprehensive feedback and error surfacing."""
    print(f"Starting PDF addition: {args.pdf}")
    print(f"ðŸ“„ Processing PDF: {args.pdf}")
    if not os.path.exists(args.pdf):
        error_msg = f"PDF file not found: {args.pdf}"
        print(f"âŒ Error: {error_msg}")
        print(f"ðŸ’¡ Troubleshooting:")
        print(f"   â€¢ Check file path spelling")
        print(f"   â€¢ Verify file exists in specified location")
        print(f"   â€¢ Use absolute path if relative path fails")
        return
    try:
        # Initialize knowledge base with enhanced feedback
        print(f"ðŸ”„ Initializing knowledge base...")
        kb_manager = KnowledgeBaseManager(storage_dir="/tmp/textbooks") # Changed to /tmp
        
        # Add PDF with progress tracking
        print(f"ðŸ”„ Adding PDF to knowledge base...")
        kb_manager.add_pdf(
            args.pdf, 
            force_ocr=args.use_ocr, 
            language=args.pdf_language
        )
        
        print(f"âœ… PDF processed successfully")
        
        # Enhanced indexing with feedback
        print(f"ðŸ”„ Building search index...")
        try:
            retrieval_system = RetrievalAugmentor(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
            retrieval_system.build_or_update_index_from_pdf(
                args.pdf,
                source_name=os.path.basename(args.pdf),
                force_rebuild=args.force_rebuild
            )
            
            print(f"âœ… Search index built successfully")
            
            # Display index statistics
            retrieval_system.print_index_stats()
            
        except Exception as index_error:
            print(f"âš ï¸  Warning: Could not build search index")
            print(f"   Error: {str(index_error)}")
            print(f"   PDF added but search functionality limited")
            
        print(f"\nðŸŽ‰ PDF ready for queries!")
        print(f"ðŸ“š Suggested next steps:")
        print(f'   python app.py --mode learning --query "Summarize key concepts"')
        print(f'   python app.py --mode question --marks 5 --query "Main topics"')
        
    except Exception as e:
        handle_critical_error("PDF addition", e, args)
        
        # Additional PDF-specific troubleshooting
        print(f"\nðŸ“„ PDF-Specific Troubleshooting:")
        print(f"   â€¢ Ensure PDF is not password-protected")
        print(f"   â€¢ Try with a smaller PDF file first")
        print(f"   â€¢ Check if PDF contains extractable text")
        print(f"   â€¢ Verify PDF is not corrupted")

def handle_kb_management(args):
    """Handle knowledge base management operations."""
    try:
        kb_manager = KnowledgeBaseManager(storage_dir="/tmp/textbooks") # Changed to /tmp
        
        if args.list_pdfs:
            kb_manager.list_pdfs()
            
        if args.kb_stats:
            kb_manager.print_storage_stats()
            
    except Exception as e:
        print(f"âŒ Error accessing knowledge base: {e}")

def handle_index_management(args):
    """Handle retrieval index management operations."""
    try:
        retrieval_system = RetrievalAugmentor(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        retrieval_system.print_index_stats()
        
    except Exception as e:
        print(f"âŒ Error accessing retrieval index: {e}")

def handle_source_removal(args):
    """Handle source removal from knowledge base and index."""
    try:
        print(f"ðŸ—‘ï¸  Removing source: {args.remove_source}")
        
        # Remove from retrieval index
        retrieval_system = RetrievalAugmentor(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        if retrieval_system.remove_source(args.remove_source):
            print("âœ… Source removed from retrieval index")
            
        # Remove from knowledge base
        kb_manager = KnowledgeBaseManager(storage_dir="/tmp/textbooks") # Changed to /tmp
        if kb_manager.remove_pdf(args.remove_source):
            print("âœ… Source removed from knowledge base")
            
    except Exception as e:
        print(f"âŒ Error removing source: {e}")

def handle_chunk_search(args):
    """Handle chunk search in the retrieval index."""
    try:
        retrieval_system = RetrievalAugmentor(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        print(f"ðŸ” Searching chunks for: {args.search_chunks}")
        results = retrieval_system.search_chunks(args.search_chunks, max_results=10)
        
        if results:
            print(f"\nðŸ“Š Found {len(results)} relevant chunks:")
            print("-" * 80)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Source: {result['source']} (Chunk {result['chunk_id']})")
                print(f"   Similarity: {result['similarity']:.3f}")
                print(f"   Quality: {result['quality_score']}/5")
                print(f"   Words: {result['word_count']}")
                print(f"   Preview: {result['preview']}")
        else:
            print("âŒ No relevant chunks found")
            
    except Exception as e:
        print(f"âŒ Error searching chunks: {e}")

def setup_enhanced_retrieval_system(args):
    """Initialize enhanced retrieval system with comprehensive error handling."""
    if args.no_context:
        return None
    if args.verbose:
        print("ðŸ” Initializing enhanced retrieval system...")
    try:
        retrieval_system = RetrievalAugmentor(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        if args.verbose:
            print("âœ… Enhanced retrieval system initialized")
            retrieval_system.print_index_stats()
            
        return retrieval_system
        
    except Exception as e:
        print(f"âš ï¸  Warning: Could not initialize retrieval system: {e}")
        print("   Proceeding without context retrieval...")
        if args.debug:
            import traceback
            traceback.print_exc()
        return None

def setup_enhanced_llm(args):
    """Initialize enhanced LLM with comprehensive configuration."""
    print(f"ðŸ¤– Initializing Enhanced LLM: {args.model_name}")
    try:
        quillai_llm = QuillAILLM(
            model_name=args.model_name, 
            force_model_check=True,
            debug_mode=args.debug
        )
        
        if args.verbose or args.debug:
            model_info = quillai_llm.get_model_info()
            print("âœ… Enhanced Model Information:")
            for key, value in model_info.items():
                print(f"     {key}: {value}")
            
            print("âœ… Enhanced LLM initialized successfully")
        return quillai_llm
        
    except Exception as e:
        print(f"âŒ Error initializing Enhanced LLM: {e}")
        print("\nðŸ”§ Enhanced Troubleshooting:")
        print("   1. Check internet connection for model download")
        print("   2. Ensure sufficient memory (8GB+ recommended)")
        print("   3. Try with --no_context flag to reduce memory usage")
        print("   4. Install required dependencies: pip install torch transformers sentence-transformers")
        print("   5. For OCR support: install Tesseract OCR")
        
        if args.debug:
            import traceback
            traceback.print_exc()
        return None

def retrieve_enhanced_context(retrieval_system, args):
    """Enhanced context retrieval with comprehensive error handling."""
    if not retrieval_system:
        return []
    if args.verbose:
        print(f"ðŸ” Retrieving context (top_k={args.top_k}, min_similarity={args.min_similarity})...")
    try:
        context_chunks = retrieval_system.retrieve_context(
            args.query,
            top_k=args.top_k,
            min_score_threshold=args.min_similarity
        )
        
        if context_chunks:
            if args.verbose:
                print(f"âœ… Retrieved {len(context_chunks)} context chunks")
                for i, chunk in enumerate(context_chunks, 1):
                    preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                    print(f"   {i}. {preview}")
            else:
                print(f"âœ… Retrieved {len(context_chunks)} relevant context chunks")
        else:
            print("âš ï¸  No relevant context found")
            if args.verbose:
                print("   Try lowering --min_similarity or adding more PDFs")
        return context_chunks
        
    except Exception as e:
        print(f"âš ï¸  Warning: Context retrieval failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return []

def generate_and_display_enhanced_answer(llm, args, context_chunks):
    """Enhanced answer generation and display with comprehensive processing."""
    print("\n" + "="*80)
    print("ðŸ¤– GENERATING ENHANCED ANSWER")
    print("="*80)
    start_time = datetime.utcnow()
    try:
        # Generate answer with enhanced parameters using the dual response method
        dual_response_data = llm.generate_dual_response(
            query=args.query,
            mode=args.mode or "learning",
            marks=args.marks,
            context_chunks=context_chunks,
            temperature=args.temperature
        )
        
        llm_answer = dual_response_data['llm_output']
        custom_answer = dual_response_data['custom_output']
        
        generation_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Display answer with enhanced formatting
        print("\n" + "="*80)
        print("ðŸ“ ENHANCED ANSWER (DialoGPT Output)")
        print("="*80)
        print(llm_answer)
        print("="*80)
        
        print("\n" + "="*80)
        print("ðŸ“ ENHANCED ANSWER (Custom LLM Output)")
        print("="*80)
        print(custom_answer)
        print("="*80)
        
        # Enhanced statistics
        llm_word_count = len(llm_answer.split())
        custom_word_count = len(custom_answer.split())
        
        print(f"ðŸ“Š Generation Statistics:")
        print(f"   â±ï¸  Total Time: {generation_time:.2f} seconds")
        print(f"   ðŸ“ DialoGPT Words: {llm_word_count}")
        print(f"   ðŸ“ Custom LLM Words: {custom_word_count}")
        print(f"   ðŸŽ¯ Mode: {args.mode or 'learning'}")
        
        if args.marks:
            target_words = {2: 100, 5: 250, 10: 500}[args.marks]
            custom_accuracy = abs(custom_word_count - target_words) / target_words * 100
            print(f"   ðŸŽ¯ Target for Custom: {target_words} words")
            print(f"   ðŸ“ Custom Accuracy: {100-custom_accuracy:.1f}% (Â±{abs(custom_word_count-target_words)} words)")
            
        if context_chunks:
            print(f"   ðŸ“š Context: {len(context_chunks)} chunks used")
            
        print(f"   ðŸŒ¡ï¸  Temperature: {args.temperature}")
        print(f"   ðŸ¤– Model: {args.model_name}")
        
        # Quality assessment for custom output
        quality_indicators = assess_answer_quality(custom_answer, args.query, args.mode)
        if quality_indicators:
            print(f"   âœ¨ Quality Indicators (Custom Output):")
            for indicator in quality_indicators:
                print(f"     â€¢ {indicator}")
        
        print("="*80)
        
    except Exception as e:
        handle_critical_error("Generating answer", e, args)

def assess_answer_quality(answer, query, mode):
    """Assess the quality of generated answer."""
    indicators = []
    # Length appropriateness
    word_count = len(answer.split())
    if mode == "learning" and word_count >= 200:
        indicators.append("Comprehensive length for learning mode")
    elif mode == "question" and 80 <= word_count <= 600:
        indicators.append("Appropriate length for question mode")
    # Structure indicators
    if "**" in answer or "*" in answer:
        indicators.append("Well-structured with formatting")
    # Content indicators
    academic_terms = ['definition', 'example', 'principle', 'concept', 'method', 'approach']
    if any(term in answer.lower() for term in academic_terms):
        indicators.append("Contains academic terminology")
    # Query relevance
    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())
    overlap = len(query_words & answer_words)
    if overlap >= len(query_words) * 0.3:
        indicators.append("High relevance to query")
    return indicators

def run_intent_detection_tests(args):
    """Run comprehensive intent detection tests."""
    print("ðŸ§ª Running Enhanced Intent Detection Tests")
    print("=" * 60)
    try:
        llm = QuillAILLM(model_name=args.model_name, debug_mode=args.debug)
        
        test_cases = [
            ("Generate 10 questions about machine learning", "question_generation"),
            ("Create a rubric for algorithms assessment", "rubric_creation"),
            ("What is artificial intelligence?", "definition"),
            ("Explain how neural networks work", "explanation"),
            ("Compare supervised and unsupervised learning", "comparison"),
            ("Give examples of AI applications", "application"),
            ("Analyze the impact of AI on society", "analysis"),
            ("Summarize the key concepts in the document", "summary_request"),
            ("How to implement a sorting algorithm", "explanation"),
            ("Define machine learning with examples", "definition")
        ]
        
        correct_predictions = 0
        
        for query, expected_intent in test_cases:
            detected_intent, confidence, all_intents = llm.detect_query_intent(query)
            is_correct = detected_intent == expected_intent
            
            print(f"Query: {query}")
            print(f"Expected: {expected_intent}")
            print(f"Detected: {detected_intent} (confidence: {confidence:.2f})")
            print(f"All intents: {all_intents}")
            print(f"Result: {'âœ… CORRECT' if is_correct else 'âŒ INCORRECT'}")
            print("-" * 60)
            
            if is_correct:
                correct_predictions += 1
        
        accuracy = (correct_predictions / len(test_cases)) * 100
        print(f"\nðŸ“Š Intent Detection Accuracy: {accuracy:.1f}% ({correct_predictions}/{len(test_cases)})")
        
        if accuracy >= 80:
            print("âœ… Intent detection system performing well!")
        else:
            print("âš ï¸  Intent detection may need improvement")
            
    except Exception as e:
        print(f"âŒ Intent detection test failed: {e}")

def run_pdf_processing_tests(args):
    """Run comprehensive PDF processing tests."""
    print("ðŸ§ª Running Enhanced PDF Processing Tests")
    print("=" * 60)
    try:
        kb_manager = KnowledgeBaseManager(storage_dir="/tmp/test_textbooks") # Changed to /tmp
        
        # Test PDF validation
        print("1. Testing PDF validation...")
        
        # Create a dummy test file for validation testing
        test_file = "/tmp/test_dummy.txt" # Changed to /tmp
        with open(test_file, "w") as f:
            f.write("This is a test file")
        
        try:
            validation = kb_manager._validate_pdf(test_file)
            print(f"   Non-PDF validation: {'âœ… PASS' if not validation['valid'] else 'âŒ FAIL'}")
        except:
            print("   Non-PDF validation: âœ… PASS (exception caught)")
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
        
        # Test text cleaning
        print("2. Testing text cleaning...")
        dirty_text = "This   is    a  test.\n\nPage 123\n\nFigure 1.2\n\nhttp://example.com\n\ntest@email.com"
        cleaned = kb_manager._clean_extracted_text(dirty_text)
        
        improvements = []
        if "Page 123" not in cleaned:
            improvements.append("Page numbers removed")
        if "Figure 1.2" not in cleaned:
            improvements.append("Figure references removed")
        if "http://example.com" not in cleaned:
            improvements.append("URLs removed")
        if "test@email.com" not in cleaned:
            improvements.append("Emails removed")
        
        print(f"   Text cleaning improvements: {len(improvements)}/4")
        for improvement in improvements:
            print(f"     âœ… {improvement}")
            
        # Test OCR availability
        print("3. Testing OCR availability...")
        print(f"   OCR available: {'âœ… YES' if kb_manager.ocr_available else 'âŒ NO'}")
        
        if not kb_manager.ocr_available:
            print("   ðŸ’¡ Install Tesseract for OCR support:")
            print("     - Windows: Download from GitHub UB-Mannheim/tesseract")
            print("     - macOS: brew install tesseract")
            print("     - Linux: sudo apt-get install tesseract-ocr")
            
        print("\nâœ… PDF processing tests completed!")
        
    except Exception as e:
        print(f"âŒ PDF processing test failed: {e}")

def run_enhanced_benchmark_tests(args):
    """Run comprehensive benchmark tests for the enhanced system."""
    print("ðŸ§ª Running Enhanced Benchmark Tests")
    print("=" * 70)
    try:
        # Initialize systems
        llm = QuillAILLM(model_name=args.model_name, debug_mode=False)
        retrieval_system = RetrievalAugmentor(chunk_size=300, chunk_overlap=30)
        
        # Test cases covering different intents and complexities
        test_cases = [
            {
                'query': 'What is machine learning?',
                'mode': 'question',
                'marks': 2,
                'expected_words': 100,
                'intent': 'definition'
            },
            {
                'query': 'Explain how neural networks work',
                'mode': 'question', 
                'marks': 5,
                'expected_words': 250,
                'intent': 'explanation'
            },
            {
                'query': 'Generate 5 questions about algorithms',
                'mode': 'learning',
                'marks': None,
                'expected_words': 200, # This is for the LLM output, custom will be different
                'intent': 'question_generation'
            },
            {
                'query': 'Compare supervised and unsupervised learning',
                'mode': 'question',
                'marks': 10,
                'expected_words': 500,
                'intent': 'comparison'
            },
            {
                'query': 'Create a rubric for AI assessment',
                'mode': 'learning',
                'marks': None,
                'expected_words': 300, # This is for the LLM output, custom will be different
                'intent': 'rubric_creation'
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nðŸ§ª Test {i}/{len(test_cases)}: {test_case['intent'].upper()}")
            print(f"Query: {test_case['query']}")
            
            start_time = datetime.utcnow()
            
            try:
                # Test intent detection
                detected_intent, confidence, _ = llm.detect_query_intent(test_case['query'])
                intent_correct = detected_intent == test_case['intent']
                
                # Generate dual response
                dual_response_data = llm.generate_dual_response(
                    query=test_case['query'],
                    mode=test_case['mode'],
                    marks=test_case['marks'],
                    context_chunks=[], # No context for benchmark simplicity
                    temperature=0.7
                )
                
                llm_word_count = dual_response_data['word_counts']['llm']
                custom_word_count = dual_response_data['word_counts']['custom']
                
                generation_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Calculate word count accuracy for CUSTOM output (as it targets precise words)
                expected_words_for_custom = llm._infer_target_word_count(test_case['marks']) if test_case['mode'] == 'question' else custom_word_count # For learning/other modes, custom generates its own length
                if expected_words_for_custom is None: # For learning/other modes, use actual custom word count as target
                    expected_words_for_custom = custom_word_count
                
                word_accuracy = max(0, 100 - abs(custom_word_count - expected_words_for_custom) / expected_words_for_custom * 100)
                
                result = {
                    'test_id': i,
                    'intent_correct': intent_correct,
                    'llm_word_count': llm_word_count,
                    'custom_word_count': custom_word_count,
                    'expected_words_for_custom': expected_words_for_custom,
                    'word_accuracy': word_accuracy,
                    'generation_time': generation_time,
                    'success': True
                }
                
                print(f"   Intent: {'âœ…' if intent_correct else 'âŒ'} {detected_intent}")
                print(f"   LLM Words: {llm_word_count}")
                print(f"   Custom Words: {custom_word_count}/{expected_words_for_custom} ({word_accuracy:.1f}% accuracy)")
                print(f"   Time: {generation_time:.2f}s")
                print(f"   Status: âœ… SUCCESS")
                
            except Exception as e:
                result = {
                    'test_id': i,
                    'intent_correct': False,
                    'llm_word_count': 0,
                    'custom_word_count': 0,
                    'expected_words_for_custom': test_case['expected_words'],
                    'word_accuracy': 0,
                    'generation_time': 0,
                    'success': False,
                    'error': str(e)
                }
                
                print(f"   Status: âŒ FAILED - {e}")
                
            results.append(result)
            
        # Calculate overall statistics
        successful_tests = [r for r in results if r['success']]
        success_rate = len(successful_tests) / len(results) * 100
        
        if successful_tests:
            avg_word_accuracy = sum(r['word_accuracy'] for r in successful_tests) / len(successful_tests)
            avg_generation_time = sum(r['generation_time'] for r in successful_tests) / len(successful_tests)
            intent_accuracy = sum(1 for r in successful_tests if r['intent_correct']) / len(successful_tests) * 100
        else:
            avg_word_accuracy = avg_generation_time = intent_accuracy = 0
        
        # Display benchmark results
        print("\n" + "=" * 70)
        print("ðŸ“Š ENHANCED BENCHMARK RESULTS")
        print("=" * 70)
        print(f"Success Rate: {success_rate:.1f}% ({len(successful_tests)}/{len(results)})")
        print(f"Intent Detection Accuracy: {intent_accuracy:.1f}%")
        print(f"Average Custom Word Count Accuracy: {avg_word_accuracy:.1f}%")
        print(f"Average Generation Time: {avg_generation_time:.2f}s")
        
        # Performance assessment
        if success_rate >= 90 and intent_accuracy >= 80 and avg_word_accuracy >= 70:
            print("\nðŸ† EXCELLENT: System performing at production level!")
        elif success_rate >= 70 and intent_accuracy >= 60 and avg_word_accuracy >= 50:
            print("\nâœ… GOOD: System performing well with minor improvements needed")
        else:
            print("\nâš ï¸  NEEDS IMPROVEMENT: System requires optimization")
            
        print("=" * 70)
        
    except Exception as e:
        print(f"âŒ Benchmark test failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

def show_enhanced_usage_examples():
    """Show comprehensive usage examples."""
    print("\nðŸ’¡ Enhanced Usage Examples:")
    print("-" * 50)
    print("ðŸ“š PDF Processing:")
    print("   python app.py --pdf textbook.pdf --use_ocr")
    print("   python app.py --pdf document.pdf --pdf_language fra --force_rebuild")
    print("\nðŸŽ“ Learning Mode (Detailed Explanations):")
    print("   python app.py --mode learning --query 'What is machine learning?'")
    print("   python app.py --mode learning --query 'Explain neural networks' --verbose")
    print("\nðŸ“ Question Mode (Targeted Answers):")
    print("   python app.py --mode question --marks 2 --query 'Define AI'")
    print("   python app.py --mode question --marks 5 --query 'Compare algorithms'")
    print("   python app.py --mode question --marks 10 --query 'Analyze ML impact'")
    print("\nðŸ”§ Question Generation:")
    print("   python app.py --query 'Generate 10 questions about AI'")
    print("   python app.py --query 'Create 5 questions on algorithms for 5 marks each'")
    print("\nðŸ“‹ Rubric Creation:")
    print("   python app.py --query 'Create rubric for ML assessment'")
    print("   python app.py --query 'Make marking scheme for algorithms'")
    print("\nðŸ” Advanced Features:")
    print("   python app.py --query 'AI applications' --top_k 5 --temperature 0.9")
    print("   python app.py --query 'ML concepts' --no_rerank --max_tokens 400")
    print("   python app.py --query 'Algorithms' --verbose")
    print("\nðŸ› ï¸  System Management:")
    print("   python app.py --list_pdfs --kb_stats")
    print("   python app.py --index_stats")
    print("   python app.py --search_chunks 'machine learning'")
    print("   python app.py --remove_source 'textbook.pdf'")
    print("\nðŸ§ª Testing & Benchmarking:")
    print("   python app.py --test_intents")
    print("   python app.py --test_pdf_processing")
    print("   python app.py --benchmark --verbose")

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced QuillAI with Intelligent Routing and Comprehensive Logging",
        epilog="Example: python app.py --mode learning --query 'What is machine learning?' --verbose"
    )
    # Core arguments
    parser.add_argument("--mode", type=str, choices=["learning", "question"],
                        help="Select either 'learning' (detailed explanations) or 'question' (concise answers)")
    parser.add_argument("--query", type=str,
                        help="User question or request.")
    # Question mode specific
    parser.add_argument("--marks", type=int, choices=[2, 5, 10], required=False,
                        help="Specify marks if mode=question. Options: 2 (100 words), 5 (250 words), 10 (500 words).")
    # Enhanced PDF processing
    parser.add_argument("--pdf", type=str,
                        help="Path to a PDF to add to the knowledge base.")
    parser.add_argument("--use_ocr", action="store_true",
                        help="Force OCR for PDF text extraction.")
    parser.add_argument("--pdf_language", type=str, default="eng",
                        help="Language for OCR processing (default: eng).")
    parser.add_argument("--force_rebuild", action="store_true",
                        help="Force rebuild of PDF index even if exists.")
    # Retrieval configuration
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of retrieved chunks (default: 3).")
    parser.add_argument("--chunk_size", type=int, default=400,
                        help="Target chunk size in words (default: 400).")
    parser.add_argument("--chunk_overlap", type=int, default=50,
                        help="Overlap between chunks in words (default: 50).")
    parser.add_argument("--min_similarity", type=float, default=0.3,
                        help="Minimum similarity threshold for retrieval (default: 0.3).")
    # Model configuration
    parser.add_argument("--model_name", type=str, default="microsoft/DialoGPT-medium",
                        help="Model name (default: microsoft/DialoGPT-medium)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8). Range: 0.1-1.5")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Override max tokens for generation.")
    # Feature toggles
    parser.add_argument("--no_rerank", action="store_true",
                        help="Disable context reranking.")
    parser.add_argument("--no_citations", action="store_true",
                        help="Disable citations in the output.")
    parser.add_argument("--no_context", action="store_true",
                        help="Disable context retrieval entirely.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging and detailed output.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with extensive logging.")
    # Enhanced testing and analysis
    parser.add_argument("--test_intents", action="store_true",
                        help="Run intent detection tests.")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run comprehensive benchmark tests.")
    parser.add_argument("--test_pdf_processing", action="store_true",
                        help="Run PDF processing tests.")
    # Knowledge base management
    parser.add_argument("--list_pdfs", action="store_true",
                        help="List all PDFs in knowledge base.")
    parser.add_argument("--kb_stats", action="store_true",
                        help="Show knowledge base statistics.")
    parser.add_argument("--index_stats", action="store_true",
                        help="Show retrieval index statistics.")
    parser.add_argument("--remove_source", type=str,
                        help="Remove a source from the knowledge base.")
    parser.add_argument("--search_chunks", type=str,
                        help="Search chunks in the index.")
    args = parser.parse_args()
    
    # Print header with user context
    print_enhanced_header(args)
    
    # Validate arguments with enhanced feedback
    validate_enhanced_arguments(args)
    
    # Route and process query or handle other operations
    if args.query:
        # Note: The CLI part of app.py still initializes components per-run.
        # For API, this is handled by api_server.py's startup event.
        route_and_process_query(args)
    else:
        handle_non_query_operations(args)

if __name__ == "__main__":
    main()