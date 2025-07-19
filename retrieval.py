import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from datetime import datetime
import json
import pickle
from typing import List, Dict, Tuple, Optional
import nltk
from collections import Counter
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Ensure NLTK data path is set to the writable /tmp directory
# This must be done before importing stopwords if they are not in default paths
if 'NLTK_DATA' in os.environ:
    nltk.data.path.append(os.environ['NLTK_DATA'])
else:
    # Fallback or default if NLTK_DATA env var is not set (e.g., local dev)
    nltk.data.path.append('/tmp/nltk_data') # Ensure this is a writable path

class RetrievalAugmentor:
    """
    Enhanced retrieval system with intelligent chunking, semantic reranking, 
    and robust error handling.
    """
    
    def __init__(self, 
                 model_name="sentence-transformers/all-MiniLM-L6-v2",
                 index_path="/tmp/faiss_index.bin", # Changed to /tmp
                 metadata_path="/tmp/metadata.json", # Changed to /tmp
                 chunk_size=400,
                 chunk_overlap=50):
        """
        Initialize enhanced retrieval system.
        
        Args:
            model_name: SentenceTransformer model name
            index_path: Path to store FAISS index
            metadata_path: Path to store chunk metadata
            chunk_size: Target chunk size in words
            chunk_overlap: Overlap between chunks in words
        """
        self.model_name = model_name
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize semantic model
        try:
            self.model = SentenceTransformer(model_name)
            print(f"âœ“ Loaded semantic model: {model_name}")
        except Exception as e:
            print(f"âŒ Failed to load semantic model: {e}")
            raise
        
        # Initialize stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
            print("âœ“ NLTK stopwords loaded successfully")
        except Exception as e:
            self.stop_words = set()
            print(f"âš  Could not load stopwords - using empty set: {e}")
            print("   Ensure NLTK data (stopwords) is downloaded to a writable location like /tmp/nltk_data")
            
        # Ensure the directory for index and metadata exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

        # Load existing index and metadata
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                print(f"âœ“ Loaded existing index with {len(self.metadata)} chunks from {self.index_path}")
            except Exception as e:
                print(f"âš  Could not load existing index from {self.index_path}: {e}")
                self._initialize_new_index()
        else:
            print(f"No existing index found at {self.index_path}. Initializing new index.")
            self._initialize_new_index()
            
        # Statistics
        self.stats = {
            'total_chunks': len(self.metadata),
            'total_queries': 0,
            'successful_retrievals': 0
        }

    def _initialize_new_index(self):
        """Initialize new empty index and metadata."""
        self.index = None
        self.metadata = []
        print("âœ“ Initialized new empty index")

    def build_or_update_index_from_pdf(self, pdf_path, source_name=None, force_rebuild=False):
        """
        Enhanced PDF indexing with intelligent chunking and comprehensive error handling.
        
        Args:
            pdf_path: Path to PDF file
            source_name: Optional name for the source (defaults to filename)
            force_rebuild: Whether to rebuild index even if source exists
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        source_name = source_name or os.path.basename(pdf_path)
        print(f"ðŸ“š Indexing PDF: {source_name}")
        
        # Check if already indexed
        if not force_rebuild:
            existing_chunks = [m for m in self.metadata if m.get('source') == source_name]
            if existing_chunks:
                print(f"âš  Source '{source_name}' already indexed ({len(existing_chunks)} chunks)")
                print("   Use force_rebuild=True to reindex")
                return
        
        try:
            # Extract and process text
            raw_text = self._extract_text_from_pdf(pdf_path)
            
            if not raw_text.strip():
                print("âŒ No text extracted from PDF")
                return
            
            print(f"âœ“ Extracted {len(raw_text):,} characters")
            
            # Clean and preprocess text
            cleaned_text = self._clean_and_preprocess_text(raw_text)
            print(f"âœ“ Cleaned text: {len(cleaned_text):,} characters")
            
            # Create intelligent chunks
            chunks = self._create_intelligent_chunks(cleaned_text, source_name)
            print(f"âœ“ Created {len(chunks)} intelligent chunks")
            
            if not chunks:
                print("âŒ No valid chunks created")
                return
            
            # Generate embeddings
            print("ðŸ”„ Generating embeddings...")
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self._generate_embeddings_batch(chunk_texts)
            
            if embeddings is None:
                print("âŒ Failed to generate embeddings")
                return
            
            # Update index
            self._update_index_with_chunks(chunks, embeddings)
            
            # Save index and metadata
            self._save_index_and_metadata()
            
            print(f"âœ… Successfully indexed {len(chunks)} chunks from {source_name}")
            
            # Update statistics
            self.stats['total_chunks'] = len(self.metadata)
            
        except Exception as e:
            error_msg = f"Failed to index PDF {source_name}: {e}"
            print(f"âŒ {error_msg}")
            raise

    def _extract_text_from_pdf(self, pdf_path):
        """
        Extract text from PDF with enhanced error handling.
        """
        text_blocks = []
        
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    try:
                        page_text = page.get_text()
                        if page_text.strip():
                            text_blocks.append(page_text)
                    except Exception as e:
                        print(f"Could not extract text from page {page_num + 1}: {e}")
                        continue
        except Exception as e:
            print(f"Could not open PDF {pdf_path}: {e}")
            raise
            
        return "\n".join(text_blocks)

    def _clean_and_preprocess_text(self, text):
        """
        Comprehensive text cleaning and preprocessing.
        """
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove headers, footers, and page numbers
        text = re.sub(r'\bpage\s+\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bchapter\s+\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bsection\s+\d+(\.\d+)*\b', '', text, flags=re.IGNORECASE)
        
        # Remove figure and table references
        text = re.sub(r'\bfigure\s+\d+(\.\d+)*\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\btable\s+\d+(\.\d+)*\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bfig\.\s*\d+\b', '', text, flags=re.IGNORECASE)
        
        # Remove repeated punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        text = re.sub(r'[=]{3,}', '===', text)
        
        # Remove standalone numbers (often artifacts)
        text = re.sub(r'\b\d+\b(?=\s|$)', '', text)
        
        # Remove URLs and email addresses (FIXED)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', text)
        
        # Normalize quotes and apostrophes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def _create_intelligent_chunks(self, text, source_name):
        """
        Create intelligent chunks using sentence boundaries and semantic coherence.
        """
        if not text.strip():
            return []
        
        chunks = []
        
        try:
            # Split into sentences
            sentences = sent_tokenize(text)
            
            if not sentences:
                # Fallback to simple splitting
                return self._create_simple_chunks(text, source_name)
            
            current_chunk = ""
            current_word_count = 0
            chunk_id = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sentence_words = len(sentence.split())
                
                # Check if adding this sentence would exceed chunk size
                if current_word_count + sentence_words > self.chunk_size and current_chunk:
                    # Save current chunk
                    if current_chunk.strip():
                        chunk_metadata = self._create_chunk_metadata(
                            current_chunk.strip(), source_name, chunk_id
                        )
                        if chunk_metadata:
                            chunks.append(chunk_metadata)
                            chunk_id += 1
                            
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0:
                        overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                        current_chunk = overlap_text + " " + sentence
                        current_word_count = len(current_chunk.split())
                    else:
                        current_chunk = sentence
                        current_word_count = sentence_words
                else:
                    # Add sentence to current chunk
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
                    current_word_count += sentence_words
            
            # Add final chunk
            if current_chunk.strip():
                chunk_metadata = self._create_chunk_metadata(
                    current_chunk.strip(), source_name, chunk_id
                )
                if chunk_metadata:
                    chunks.append(chunk_metadata)
                    
        except Exception as e:
            print(f"Intelligent chunking failed: {e}, falling back to simple chunking")
            return self._create_simple_chunks(text, source_name)
            
        return chunks

    def _create_simple_chunks(self, text, source_name):
        """
        Fallback simple chunking method.
        """
        words = text.split()
        chunks = []
        chunk_id = 0
        
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            
            chunk_metadata = self._create_chunk_metadata(chunk_text, source_name, chunk_id)
            if chunk_metadata:
                chunks.append(chunk_metadata)
                chunk_id += 1
            
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end
            
        return chunks

    def _get_overlap_text(self, text, overlap_words):
        """
        Get the last N words from text for overlap.
        """
        words = text.split()
        if len(words) <= overlap_words:
            return text
        return " ".join(words[-overlap_words:])

    def _create_chunk_metadata(self, text, source_name, chunk_id):
        """
        Create metadata for a text chunk with quality checks.
        """
        if not text or len(text.strip()) < 20:  # Skip very short chunks
            return None
        
        # Calculate basic statistics
        word_count = len(text.split())
        char_count = len(text)
        
        # Quality checks
        if word_count < 10:  # Skip chunks that are too short
            return None
        
        # Check for meaningful content (not just numbers/symbols)
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.5:  # Skip chunks with too few letters
            return None
        
        return {
            'text': text,
            'source': source_name,
            'chunk_id': chunk_id,
            'word_count': word_count,
            'char_count': char_count,
            'created_at': datetime.utcnow().isoformat(),
            'quality_score': self._calculate_quality_score(text)
        }

    def _calculate_quality_score(self, text):
        """
        Calculate a quality score for a text chunk.
        """
        score = 0
        
        # Length score (prefer medium-length chunks)
        word_count = len(text.split())
        if 50 <= word_count <= 300:
            score += 2
        elif 20 <= word_count < 50 or 300 < word_count <= 500:
            score += 1
            
        # Academic content indicators
        academic_terms = [
            'algorithm', 'method', 'approach', 'technique', 'process',
            'system', 'model', 'theory', 'principle', 'concept',
            'definition', 'example', 'application', 'analysis'
        ]
        
        text_lower = text.lower()
        academic_score = sum(1 for term in academic_terms if term in text_lower)
        score += min(academic_score, 3)  # Cap at 3 points
        
        # Sentence structure (prefer complete sentences)
        sentence_count = len([s for s in text.split('.') if s.strip()])
        if sentence_count >= 2:
            score += 1
            
        # Avoid repetitive content
        words = text.lower().split()
        unique_words = set(words)
        if len(unique_words) / len(words) > 0.7:  # Good word diversity
            score += 1
            
        return score

    def _generate_embeddings_batch(self, texts, batch_size=32):
        """
        Generate embeddings for texts in batches with error handling.
        """
        try:
            embeddings = self.model.encode(
                texts, 
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=True
            )
            return embeddings
        except Exception as e:
            print(f"Failed to generate embeddings: {e}")
            return None

    def _update_index_with_chunks(self, chunks, embeddings):
        """
        Update FAISS index with new chunks and embeddings.
        """
        if self.index is None:
            # Create new index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            print(f"âœ“ Created new FAISS index (dimension: {dimension})")
            
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Add chunks to metadata
        self.metadata.extend(chunks)
        
        print(f"âœ“ Added {len(chunks)} chunks to index")

    def _save_index_and_metadata(self, ):
        """
        Save FAISS index and metadata to disk.
        """
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
            print(f"âœ“ Saved index and metadata to {self.index_path} and {self.metadata_path}")
            
        except Exception as e:
            print(f"Failed to save index/metadata: {e}")
            raise

    def retrieve_context(self, query, top_k=5, min_score_threshold=0.3):
        """
        Enhanced context retrieval with semantic reranking and quality filtering.
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            min_score_threshold: Minimum similarity score threshold
        """
        if self.index is None or not self.metadata:
            print("âš  No index available for retrieval")
            return []
        
        self.stats['total_queries'] += 1
        
        try:
            print(f"ðŸ” Retrieving context for: {query[:50]}...")
            
            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # Search in FAISS index
            # Get more candidates than needed for reranking
            search_k = min(top_k * 3, len(self.metadata))
            distances, indices = self.index.search(query_embedding, search_k)
            
            # Convert distances to similarity scores (FAISS uses L2 distance)
            similarities = 1 / (1 + distances[0])
            
            # Get candidate chunks with scores
            candidates = []
            for idx, similarity in zip(indices[0], similarities):
                if 0 <= idx < len(self.metadata):
                    chunk = self.metadata[idx]
                    candidates.append({
                        'chunk': chunk,
                        'similarity': similarity,
                        'index': idx
                    })
            
            # Filter by minimum score threshold
            candidates = [c for c in candidates if c['similarity'] >= min_score_threshold]
            
            if not candidates:
                print(f"âš  No chunks found above similarity threshold {min_score_threshold}")
                return []
            
            # Enhanced reranking
            reranked_candidates = self._enhanced_rerank_candidates(query, candidates)
            
            # Select top results
            top_candidates = reranked_candidates[:top_k]
            
            # Extract chunk texts
            relevant_chunks = [c['chunk']['text'] for c in top_candidates]
            
            print(f"âœ“ Retrieved {len(relevant_chunks)} relevant chunks")
            
            # Log retrieval details
            if relevant_chunks:
                self.stats['successful_retrievals'] += 1
                avg_similarity = np.mean([c['similarity'] for c in top_candidates])
                
            return relevant_chunks
            
        except Exception as e:
            error_msg = f"Context retrieval failed: {e}"
            print(f"âŒ {error_msg}")
            return []

    def _enhanced_rerank_candidates(self, query, candidates):
        """
        Enhanced reranking using multiple signals.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Remove stopwords from query
        query_words_filtered = query_words - self.stop_words
        
        for candidate in candidates:
            chunk_text = candidate['chunk']['text'].lower()
            chunk_words = set(chunk_text.split())
            
            # Base similarity score
            base_score = candidate['similarity']
            
            # Keyword overlap bonus
            keyword_overlap = len(query_words_filtered & chunk_words)
            keyword_bonus = keyword_overlap * 0.1
            
            # Quality score bonus
            quality_bonus = candidate['chunk'].get('quality_score', 0) * 0.05
            
            # Length penalty for very short or very long chunks
            word_count = candidate['chunk']['word_count']
            length_penalty = 0
            if word_count < 30:
                length_penalty = -0.2
            elif word_count > 400:
                length_penalty = -0.1
            
            # Academic content bonus
            academic_terms = [
                'algorithm', 'method', 'approach', 'technique', 'process',
                'definition', 'example', 'principle', 'concept'
            ]
            academic_bonus = sum(0.02 for term in academic_terms if term in chunk_text)
            
            # Calculate final score
            final_score = base_score + keyword_bonus + quality_bonus + length_penalty + academic_bonus
            candidate['final_score'] = final_score
            
        # Sort by final score
        return sorted(candidates, key=lambda x: x['final_score'], reverse=True)

    def get_index_stats(self):
        """
        Get comprehensive statistics about the index.
        """
        if not self.metadata:
            return {
                'total_chunks': 0,
                'total_sources': 0,
                'sources': [],
                'index_size': 0,
                'avg_chunk_words': 0,
                'avg_quality_score': 0,
                'total_queries': self.stats['total_queries'],
                'successful_retrievals': self.stats['successful_retrievals'],
                'success_rate': 0
            }
        
        # Count sources
        sources = set(chunk.get('source', 'unknown') for chunk in self.metadata)
        
        # Calculate statistics
        word_counts = [chunk.get('word_count', 0) for chunk in self.metadata]
        quality_scores = [chunk.get('quality_score', 0) for chunk in self.metadata]
        
        stats = {
            'total_chunks': len(self.metadata),
            'total_sources': len(sources),
            'sources': list(sources),
            'index_size': self.index.ntotal if self.index else 0,
            'avg_chunk_words': np.mean(word_counts) if word_counts else 0,
            'avg_quality_score': np.mean(quality_scores) if quality_scores else 0,
            'total_queries': self.stats['total_queries'],
            'successful_retrievals': self.stats['successful_retrievals'],
            'success_rate': (self.stats['successful_retrievals'] / max(self.stats['total_queries'], 1)) * 100
        }
        
        return stats

    def print_index_stats(self):
        """
        Print comprehensive index statistics.
        """
        stats = self.get_index_stats()
        
        print("ðŸ“Š Retrieval Index Statistics:")
        print("-" * 40)
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Total sources: {stats['total_sources']}")
        print(f"Index size: {stats['index_size']}")
        print(f"Average chunk words: {stats['avg_chunk_words']:.1f}")
        print(f"Average quality score: {stats['avg_quality_score']:.1f}")
        print(f"Total queries: {stats['total_queries']}")
        print(f"Successful retrievals: {stats['successful_retrievals']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        
        if stats['sources']:
            print(f"\nSources:")
            for source in stats['sources']:
                source_chunks = [c for c in self.metadata if c.get('source') == source]
                print(f"   â€¢ {source}: {len(source_chunks)} chunks")

    def search_chunks(self, query, max_results=10):
        """
        Search chunks and return detailed results with metadata.
        """
        if self.index is None or not self.metadata:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # Search
            search_k = min(max_results * 2, len(self.metadata))
            distances, indices = self.index.search(query_embedding, search_k)
            
            # Prepare results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if 0 <= idx < len(self.metadata):
                    chunk = self.metadata[idx]
                    similarity = 1 / (1 + distance)
                    
                    results.append({
                        'text': chunk['text'],
                        'source': chunk.get('source', 'unknown'),
                        'chunk_id': chunk.get('chunk_id', 0),
                        'word_count': chunk.get('word_count', 0),
                        'quality_score': chunk.get('quality_score', 0),
                        'similarity': similarity,
                        'preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                    })
            
            # Sort by similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            return results[:max_results]
            
        except Exception as e:
            print(f"Chunk search failed: {e}")
            return []
            
    def remove_source(self, source_name):
        """
        Remove all chunks from a specific source.
        """
        if not self.metadata:
            print(f"No chunks to remove for source: {source_name}")
            return False
        
        # Find chunks to remove
        chunks_to_remove = [i for i, chunk in enumerate(self.metadata) if chunk.get('source') == source_name]
        
        if not chunks_to_remove:
            print(f"No chunks found for source: {source_name}")
            return False
        
        print(f"ðŸ—‘ï¸  Removing {len(chunks_to_remove)} chunks from source: {source_name}")
        
        try:
            # Remove chunks from metadata (in reverse order to maintain indices)
            for i in reversed(chunks_to_remove):
                del self.metadata[i]
            
            # Rebuild index (FAISS doesn't support efficient deletion)
            if self.metadata:
                print("ðŸ”„ Rebuilding index...")
                chunk_texts = [chunk['text'] for chunk in self.metadata]
                embeddings = self._generate_embeddings_batch(chunk_texts)
                
                if embeddings is not None:
                    # Create new index
                    dimension = embeddings.shape[1]
                    self.index = faiss.IndexFlatL2(dimension)
                    self.index.add(embeddings)
                    
                    # Save updated index and metadata
                    self._save_index_and_metadata()
                    
                    print(f"âœ… Successfully removed source: {source_name}")
                    self.stats['total_chunks'] = len(self.metadata)
                    return True
                else:
                    print("âŒ Failed to rebuild index")
                    return False
            else:
                # No chunks left, reset index
                self._initialize_new_index()
                self._save_index_and_metadata()
                print(f"âœ… Removed last source: {source_name}")
                self.stats['total_chunks'] = 0
                return True
                
        except Exception as e:
            error_msg = f"Failed to remove source {source_name}: {e}"
            print(f"âŒ {error_msg}")
            return False

# Example usage and testing
if __name__ == "__main__":
    print("Enhanced Retrieval Augmentor Test")
    print("=" * 50)
    
    # Initialize retrieval system
    retrieval = RetrievalAugmentor(chunk_size=300, chunk_overlap=30)
    
    # Print current stats
    retrieval.print_index_stats()
    
    print("\nRetrieval Augmentor ready for use!")
    print("Example usage:")
    print("   retrieval.build_or_update_index_from_pdf('textbook.pdf')")
    print("   chunks = retrieval.retrieve_context('machine learning', top_k=3)")
    print("   results = retrieval.search_chunks('algorithms', max_results=5)")